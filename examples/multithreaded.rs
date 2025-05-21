//! randomx example that calculates many hashes using multiple threads

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;

use anyhow::Result;
use randomx::*;

#[derive(Clone)]
pub struct RandomXVMInstance {
    instance: Arc<RwLock<RandomXVM>>,
}

unsafe impl Send for RandomXVMInstance {}
unsafe impl Sync for RandomXVMInstance {}

impl RandomXVMInstance {
    fn create(
        key: &[u8],
        flags: RandomXFlags,
        cache: Option<RandomXCache>,
        dataset: Option<RandomXDataset>,
    ) -> Result<Self> {
        // Note: Memory requirement per VM in light mode is 256MB
        // Note: RandomXFlags::FULLMEM and RandomXFlags::LARGEPAGES are incompatible
        // with light mode. These are not set by RandomX automatically even in fast mode.
        let (flags, cache) = match cache {
            Some(c) => (flags, c),
            None => match RandomXCache::new(flags, key) {
                Ok(cache) => (flags, cache),
                Err(_) => {
                    // Fallback to default flags
                    let flags = RandomXFlags::DEFAULT;
                    let cache = RandomXCache::new(flags, key)?;
                    (flags, cache)
                }
            },
        };

        let vm = RandomXVM::new(flags, Some(cache), dataset)?;

        Ok(Self {
            instance: Arc::new(RwLock::new(vm)),
        })
    }

    /// Calculate the RandomX mining hash
    pub fn calculate_hash(&self, input: &[u8]) -> Result<Vec<u8>> {
        let lock = self.instance.write().unwrap();
        Ok(lock.calculate_hash(input)?)
    }
}

#[derive(Clone, Debug)]
pub struct RandomXFactory {
    inner: Arc<RwLock<RandomXFactoryInner>>,
}

impl Default for RandomXFactory {
    fn default() -> Self {
        Self::new(2)
    }
}

impl RandomXFactory {
    /// Create a new RandomX factory with the specified maximum number of VMs
    pub fn new(max_vms: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(RandomXFactoryInner::new(max_vms))),
        }
    }

    pub fn new_with_flags(max_vms: usize, flags: RandomXFlags) -> Self {
        Self {
            inner: Arc::new(RwLock::new(RandomXFactoryInner::new_with_flags(
                max_vms, flags,
            ))),
        }
    }

    /// Create a new RandomX VM instance with the specified key
    pub fn create(
        &self,
        key: &[u8],
        cache: Option<RandomXCache>,
        dataset: Option<RandomXDataset>,
    ) -> Result<RandomXVMInstance> {
        let res;
        {
            let mut inner = self.inner.write().unwrap();
            res = inner.create(key, cache, dataset)?;
        }
        Ok(res)
    }

    /// Get the number of VMs currently allocated
    pub fn get_count(&self) -> Result<usize> {
        let inner = self.inner.read().unwrap();
        Ok(inner.get_count())
    }

    /// Get the flags used to create the VMs
    pub fn get_flags(&self) -> Result<RandomXFlags> {
        let inner = self.inner.read().unwrap();
        Ok(inner.get_flags())
    }
}

struct RandomXFactoryInner {
    flags: RandomXFlags,
    vms: HashMap<Vec<u8>, (Instant, RandomXVMInstance)>,
    max_vms: usize,
}

impl std::fmt::Debug for RandomXFactoryInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomXFactory")
            .field("flags", &self.flags)
            .field("max_vms", &self.max_vms)
            .finish()
    }
}

impl RandomXFactoryInner {
    fn new(max_vms: usize) -> Self {
        let flags = RandomXFlags::get_recommended_flags();
        Self {
            flags,
            vms: Default::default(),
            max_vms,
        }
    }

    fn new_with_flags(max_vms: usize, flags: RandomXFlags) -> Self {
        Self {
            flags,
            vms: Default::default(),
            max_vms,
        }
    }

    fn create(
        &mut self,
        key: &[u8],
        cache: Option<RandomXCache>,
        dataset: Option<RandomXDataset>,
    ) -> Result<RandomXVMInstance> {
        if let Some(entry) = self.vms.get_mut(key) {
            let vm = entry.1.clone();
            entry.0 = Instant::now();
            return Ok(vm);
        }

        if self.vms.len() >= self.max_vms {
            if let Some(oldest_key) = self
                .vms
                .iter()
                .min_by_key(|(_, (i, _))| *i)
                .map(|(k, _)| k.clone())
            {
                self.vms.remove(&oldest_key);
            }
        }

        let vm = RandomXVMInstance::create(key, self.flags, cache, dataset)?;

        self.vms
            .insert(Vec::from(key), (Instant::now(), vm.clone()));

        Ok(vm)
    }

    /// Get the number of VMs currently allocated
    fn get_count(&self) -> usize {
        self.vms.len()
    }

    /// Get the flags used to create the VMs
    fn get_flags(&self) -> RandomXFlags {
        self.flags
    }
}

fn main() {
    const NUM_THREADS: u32 = 8;
    // number of hashes to perform in each thread, not the total.
    const NUM_HASHES: u32 = 10000;

    // Try adding `| RandomXFlags::LARGEPAGES`.
    let mut flags = RandomXFlags::get_recommended_flags() | RandomXFlags::FULLMEM;
    if is_x86_feature_detected!("avx2") {
        flags |= RandomXFlags::ARGON2_AVX2;
    } else if is_x86_feature_detected!("ssse3") {
        flags |= RandomXFlags::ARGON2_SSSE3;
    }

    let factory = RandomXFactory::new_with_flags(8, flags);

    let key = b"key";

    let start = Instant::now();
    let cache = RandomXCache::new(flags, &key[..]).unwrap();
    let dataset_item_count = RandomXDataset::count().unwrap();
    println!("Initialized RandomX cache in {:?}", start.elapsed());

    let mut handles = Vec::new();
    let start = Instant::now();

    for i in 0..NUM_THREADS {
        let factory = factory.clone();

        let ds_start = Instant::now();
        let dataset = if NUM_THREADS > 1 {
            let a = (dataset_item_count * i) / NUM_THREADS;
            let b = (dataset_item_count * (i + 1)) / NUM_THREADS;
            /*
            println!("a={a}");
            println!("b={b}");
            println!("b-a={}", b - a);
            */
            RandomXDataset::new(flags, cache.clone(), a, b - a).unwrap()
        } else {
            RandomXDataset::new(flags, cache.clone(), 0, dataset_item_count).unwrap()
        };
        println!(
            "Initialized RandomX dataset for thread #{i} in {:?}",
            ds_start.elapsed()
        );

        handles.push(thread::spawn(move || {
            let key = b"key";
            let vm = factory.create(&key[..], None, Some(dataset)).unwrap();
            println!("Initialized RandomX VM #{i}");

            let mut nonce: u32 = i;

            for _ in 0..NUM_HASHES {
                let _ = vm.calculate_hash(&nonce.to_be_bytes()[..]);
                //println!("VM #{i} calculated hash with nonce {nonce}");

                // e.g. thread 0 will use nonces 0, 8, 16, ...
                // and thread 1 will use nonces 1, 9, 17, ...
                nonce += NUM_THREADS;
            }
        }));
    }

    for handle in handles {
        let _ = handle.join();
    }

    println!(
        "Completed {} hashes in {:?}",
        NUM_THREADS * NUM_HASHES,
        start.elapsed()
    );
}
