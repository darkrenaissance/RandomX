#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

//! Rust bindings to randomx, a library for computing RandomX hashes.
//!
//! "RandomX is a proof-of-work (PoW) algorithm that is optimized for general-purpose CPUs. RandomX uses random code
//! execution together with several memory-hard techniques to minimize the efficiency advantage of specialized
//! hardware."
//!
//! Read more about how RandomX works in the [design document].
//!
//! [RandomX github repo]: <https://github.com/tevador/RandomX>
//! [design document]: <https://github.com/tevador/RandomX/blob/master/doc/design.md>
//!
//! # Errors
//!
//! Some operations (e.g. allocating a VM or dataset) can fail if the
//! system doesn't have enough free memory, or if you tried to force a
//! feature like large pages or AVX2 on a system that does not support
//! it.

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ptr;
use std::sync::Arc;

use bitflags::bitflags;
use libc::{c_ulong, c_void, memcpy};

#[derive(Debug, Clone)]
pub enum RandomXError {
    /// Occurs when allocating the RandomX cache fails.
    ///
    /// Reasons include:
    /// * Memory allocation fails
    /// * The JIT flag is set but the current platform does not support it
    /// * An invalid or unsupported ARGON2 value is set
    CacheAllocError,

    /// Occurs when RandomX cache being reinitialized fails.
    ///
    /// Reasons include:
    /// * VM is initialized with FULLMEM flag set.
    CacheReinitError,

    /// Occurs when allocating a RandomX dataset fails.
    ///
    /// Reasons include:
    /// * Memory allocation fails
    DatasetAllocError,

    /// Occurs when RandomX dataset being reinitialized fails.
    ///
    /// Reasons include:
    /// * VM is initialized without FULLMEM flag set.
    DatasetReinitError,

    /// Occurs when creating a VM fails.
    ///
    /// Reasons included:
    /// * Scratchpad memory allocation fails
    /// * Unsupported flags
    VmAllocError,

    /// Various parameter errors; self-explanatory
    ParameterError(String),

    /// Other errors
    Other(String),
}

impl std::error::Error for RandomXError {}

impl std::fmt::Display for RandomXError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::CacheAllocError => write!(f, "Failed to allocate RandomX cache"),
            Self::CacheReinitError => write!(f, "Can't reinit RandomX cache w/ FULLMEM set"),
            Self::DatasetAllocError => write!(f, "Failed to allocate RandomX dataset"),
            Self::DatasetReinitError => write!(f, "Can't reinit RandomX dataset w/o FULLMEM set"),
            Self::VmAllocError => write!(f, "Failed to allocate RandomX VM"),
            Self::ParameterError(e) => write!(f, "{e}"),
            Self::Other(e) => write!(f, "{e}"),
        }
    }
}

bitflags! {
    /// Represents options that can be used when allocating the
    /// RandomX dataset or VM.
    #[derive(Debug, Copy, Clone)]
    pub struct RandomXFlags: u32 {
        /// Use defaults.
        const DEFAULT = randomx_flags_RANDOMX_FLAG_DEFAULT;

        /// Allocate memory in large pages.
        const LARGEPAGES = randomx_flags_RANDOMX_FLAG_LARGE_PAGES;

        /// The RandomX VM will use hardware accelerated AES.
        const HARDAES = randomx_flags_RANDOMX_FLAG_HARD_AES;

        /// The RandomX VM will use the full dataset.
        const FULLMEM = randomx_flags_RANDOMX_FLAG_FULL_MEM;

        /// The RandomX VM will use a JIT compiler.
        const JIT = randomx_flags_RANDOMX_FLAG_JIT;

        /// Make sure that JIT pages are never writable and executable
        /// at the same time.
        const SECURE = randomx_flags_RANDOMX_FLAG_SECURE;

        /// Use the SSSE3 extension to speed up Argon2 operations.
        const ARGON2_SSSE3 = randomx_flags_RANDOMX_FLAG_ARGON2_SSSE3;

        /// Use the AVX2 extension to speed up Argon2 operations.
        const ARGON2_AVX2 = randomx_flags_RANDOMX_FLAG_ARGON2_AVX2;

        /// Do not use SSSE3 or AVX2 extensions.
        const ARGON2 = randomx_flags_RANDOMX_FLAG_ARGON2;
    }
}

impl RandomXFlags {
    /// Returns the recommended flags to be used.
    ///
    /// Does not include:
    /// * LARGEPAGES
    /// * FULLMEM
    /// * SECURE
    ///
    /// The above flags need to be set manually, if required.
    pub fn get_recommended_flags() -> Self {
        unsafe { Self::from_bits(randomx_get_flags()).unwrap() }
    }
}

impl Default for RandomXFlags {
    /// Default value for RandomXFlags
    fn default() -> RandomXFlags {
        RandomXFlags::DEFAULT
    }
}

#[derive(Debug)]
struct RandomXCacheInner {
    cache_ptr: *mut randomx_cache,
}

unsafe impl Send for RandomXCacheInner {}
unsafe impl Sync for RandomXCacheInner {}

impl Drop for RandomXCacheInner {
    /// Deallocates memory for the `cache` object
    fn drop(&mut self) {
        unsafe {
            randomx_release_cache(self.cache_ptr);
        }
    }
}

#[derive(Debug, Clone)]
/// The Cache is used for light verification and Dataset construction.
pub struct RandomXCache {
    inner: Arc<RandomXCacheInner>,
}

impl RandomXCache {
    /// Creates and allocates memory for a new cache object, and initializes
    /// it with the key value.
    ///
    /// `flags` is any combination of the following two flags:
    /// * LARGEPAGES
    /// * JIT
    ///
    /// and (optionally) one of the following flags (depending on instruction set)
    /// * ARGON2_SSSE3
    /// * ARGON2_AVX2
    ///
    /// `key` is a sequence of u8 used to initialize SuperScalarHash.
    pub fn new(flags: RandomXFlags, key: &[u8]) -> Result<RandomXCache, RandomXError> {
        if key.is_empty() {
            return Err(RandomXError::ParameterError(
                "RandomX cache key is empty".to_string(),
            ));
        }

        let cache_ptr = unsafe { randomx_alloc_cache(flags.bits()) };
        if cache_ptr.is_null() {
            return Err(RandomXError::CacheAllocError);
        }

        let inner = RandomXCacheInner { cache_ptr };
        let result = RandomXCache {
            inner: Arc::new(inner),
        };
        let key_ptr = key.as_ptr() as *mut c_void;
        let key_size = key.len();

        unsafe {
            randomx_init_cache(result.inner.cache_ptr, key_ptr, key_size);
        }

        Ok(result)
    }
}

#[derive(Debug)]
struct RandomXDatasetInner {
    dataset_ptr: *mut randomx_dataset,
    dataset_count: u32,
    #[allow(dead_code)]
    cache: RandomXCache,
}

unsafe impl Send for RandomXDatasetInner {}
unsafe impl Sync for RandomXDatasetInner {}

impl Drop for RandomXDatasetInner {
    /// Deallocates memory for the `dataset` object.
    fn drop(&mut self) {
        unsafe {
            randomx_release_dataset(self.dataset_ptr);
        }
    }
}

#[derive(Debug, Clone)]
/// The Dataset is a read-only memory structure that is used during
/// VM program execution.
pub struct RandomXDataset {
    inner: Arc<RandomXDatasetInner>,
}

impl RandomXDataset {
    /// Creates a new dataset object, allocates memory to the `dataset` object
    /// and initializes it.
    ///
    /// `flags` is one of the following:
    /// * DEFAULT
    /// * LARGEPAGES
    ///
    /// `cache` is a cache object.
    ///
    /// `start_item` is the item number where initialization should start,
    /// recommended to pass in 0.
    ///
    /// `item_count` is the total item count in the dataset, it can be
    /// retrieved with `RandomXDataset::count()`.
    ///
    /// Conversions may be lossy on Windows or Linux.
    #[allow(clippy::useless_conversion)]
    pub fn new(
        flags: RandomXFlags,
        cache: RandomXCache,
        start_item: u32,
        item_count: u32,
    ) -> Result<RandomXDataset, RandomXError> {
        let test = unsafe { randomx_alloc_dataset(flags.bits()) };
        if test.is_null() {
            return Err(RandomXError::DatasetAllocError);
        }

        let inner = RandomXDatasetInner {
            dataset_ptr: test,
            dataset_count: item_count,
            cache,
        };

        let result = RandomXDataset {
            inner: Arc::new(inner),
        };

        if start_item >= item_count {
            return Err(RandomXError::DatasetAllocError);
        }

        unsafe {
            randomx_init_dataset(
                result.inner.dataset_ptr,
                result.inner.cache.inner.cache_ptr,
                c_ulong::from(start_item),
                c_ulong::from(item_count),
            );
        }

        Ok(result)
    }

    /// Returns the number of items in the `dataset` or an error on failure.
    pub fn count() -> Result<u32, RandomXError> {
        match unsafe { randomx_dataset_item_count() } {
            0 => Err(RandomXError::ParameterError(
                "Dataset item count is zero".to_string(),
            )),
            x => {
                // This weirdness brought to you by c_ulong being different on Windows and Linux
                #[cfg(target_os = "windows")]
                return Ok(x);
                #[cfg(not(target_os = "windows"))]
                return u32::try_from(x).map_err(|e| RandomXError::Other(e.to_string()));
            }
        }
    }

    /// Returns the values of the internal memory buffer of the `dataset` or an error on failure.
    pub fn get_data(&self) -> Result<Vec<u8>, RandomXError> {
        let memory = unsafe { randomx_get_dataset_memory(self.inner.dataset_ptr) };
        if memory.is_null() {
            return Err(RandomXError::DatasetAllocError);
        }

        let count = usize::try_from(self.inner.dataset_count)
            .map_err(|e| RandomXError::Other(e.to_string()))?;

        let mut result: Vec<u8> = vec![0u8; count];

        let n = usize::try_from(self.inner.dataset_count)
            .map_err(|e| RandomXError::Other(e.to_string()))?;

        unsafe {
            memcpy(result.as_mut_ptr() as *mut c_void, memory, n);
        }

        Ok(result)
    }
}

#[derive(Debug)]
/// The RandomX Virtual Machine (VM) is a complex instruction set computer
/// that executes generated programs.
pub struct RandomXVM {
    flags: RandomXFlags,
    vm: *mut randomx_vm,
    linked_cache: Option<RandomXCache>,
    linked_dataset: Option<RandomXDataset>,
}

unsafe impl Send for RandomXVM {}
unsafe impl Sync for RandomXVM {}

impl Drop for RandomXVM {
    /// De-allocates memory for the `VM` object.
    fn drop(&mut self) {
        unsafe {
            randomx_destroy_vm(self.vm);
        }
    }
}

impl RandomXVM {
    /// Creates a new `VM` and initializes it, error on failure.
    ///
    /// `flags` is any combination of the following 5 flags:
    /// * LARGEPAGES
    /// * HARDAES
    /// * FULLMEM
    /// * JIT
    /// * SECURE
    ///
    /// Or
    ///
    /// * DEFAULT
    ///
    /// `cache` is a cache object, optional if FULLMEM is set.
    ///
    /// `dataset` is a dataset object, optional if FULLMEM is not set.
    pub fn new(
        flags: RandomXFlags,
        cache: Option<RandomXCache>,
        dataset: Option<RandomXDataset>,
    ) -> Result<RandomXVM, RandomXError> {
        let is_full_mem = flags.contains(RandomXFlags::FULLMEM);

        match (cache, dataset) {
            (None, None) => Err(RandomXError::VmAllocError),
            (None, _) if !is_full_mem => Err(RandomXError::VmAllocError),
            (_, None) if is_full_mem => Err(RandomXError::VmAllocError),
            (cache, dataset) => {
                let cache_ptr = cache
                    .as_ref()
                    .map(|stash| stash.inner.cache_ptr)
                    .unwrap_or_else(ptr::null_mut);
                let dataset_ptr = dataset
                    .as_ref()
                    .map(|data| data.inner.dataset_ptr)
                    .unwrap_or_else(ptr::null_mut);

                let vm = unsafe { randomx_create_vm(flags.bits(), cache_ptr, dataset_ptr) };

                Ok(RandomXVM {
                    vm,
                    flags,
                    linked_cache: cache,
                    linked_dataset: dataset,
                })
            }
        }
    }

    /// Reinitializes the `VM` with a new cache that was initialized without
    /// `RandomXFlags::FULLMEM`.
    pub fn reinit_cache(&mut self, cache: RandomXCache) -> Result<(), RandomXError> {
        if self.flags.contains(RandomXFlags::FULLMEM) {
            return Err(RandomXError::CacheReinitError);
        }

        unsafe {
            randomx_vm_set_cache(self.vm, cache.inner.cache_ptr);
        }

        self.linked_cache = Some(cache);

        Ok(())
    }

    /// Reinitializes the `VM` with a new dataset that was initialized with
    /// `RandomXFlags::FULLMEM`.
    pub fn reinit_dataset(&mut self, dataset: RandomXDataset) -> Result<(), RandomXError> {
        if !self.flags.contains(RandomXFlags::FULLMEM) {
            return Err(RandomXError::DatasetReinitError);
        }

        unsafe {
            randomx_vm_set_dataset(self.vm, dataset.inner.dataset_ptr);
        }

        self.linked_dataset = Some(dataset);

        Ok(())
    }

    /// Calculates a RandomX hash value and returns it, error on failure.
    ///
    /// `input` is a sequence of u8 to be hashed.
    pub fn calculate_hash(&self, input: &[u8]) -> Result<Vec<u8>, RandomXError> {
        if input.is_empty() {
            return Err(RandomXError::ParameterError(
                "RandomX VM input empty".to_string(),
            ));
        }

        let input_size = input.len();
        let input_ptr = input.as_ptr() as *mut c_void;
        let arr = [0; RANDOMX_HASH_SIZE as usize];
        let output_ptr = arr.as_ptr() as *mut c_void;

        unsafe {
            randomx_calculate_hash(self.vm, input_ptr, input_size, output_ptr);
        }

        // If this failed, arr should still be empty
        if arr == [0; RANDOMX_HASH_SIZE as usize] {
            return Err(RandomXError::Other(
                "RandomX calculated hash was empty".to_string(),
            ));
        }

        Ok(arr.to_vec())
    }

    /// Calculates hashes from a set of inputs.
    ///
    /// `input` is an array of a sequence of u8 to be hashed.
    pub fn calculate_hash_set(&self, input: &[&[u8]]) -> Result<Vec<Vec<u8>>, RandomXError> {
        if input.is_empty() {
            // Empty set
            return Err(RandomXError::ParameterError(
                "RandomX VM input set empty".to_string(),
            ));
        }

        let mut result = vec![];

        // For single input
        if input.len() == 1 {
            let hash = self.calculate_hash(input[0])?;
            result.push(hash);
            return Ok(result);
        }

        // For multiple inputs
        let mut output_ptr: *mut c_void = ptr::null_mut();
        let arr = [0; RANDOMX_HASH_SIZE as usize];

        // Not len() as last iteration assigns final hash
        let iterations = input.len() + 1;

        #[allow(clippy::needless_range_loop)]
        for i in 0..iterations {
            if i == iterations - 1 {
                // For last iteration
                unsafe {
                    randomx_calculate_hash_last(self.vm, output_ptr);
                }
            } else {
                if input[i].is_empty() {
                    // Stop calculations
                    if arr != [0; RANDOMX_HASH_SIZE as usize] {
                        // Complete what was started
                        unsafe {
                            randomx_calculate_hash_last(self.vm, output_ptr);
                        }
                    }
                    return Err(RandomXError::ParameterError(
                        "RandomX VM input was empty".to_string(),
                    ));
                }

                let input_size = input[i].len();
                let input_ptr = input[i].as_ptr() as *mut c_void;
                output_ptr = arr.as_ptr() as *mut c_void;

                if i == 0 {
                    // For first iteration
                    unsafe {
                        randomx_calculate_hash_first(self.vm, input_ptr, input_size);
                    }
                } else {
                    // For every other iteration
                    unsafe {
                        randomx_calculate_hash_next(self.vm, input_ptr, input_size, output_ptr);
                    }
                }
            }

            if i != 0 {
                // First hash is only available in 2nd iteration
                if arr == [0; RANDOMX_HASH_SIZE as usize] {
                    return Err(RandomXError::Other("RandomX hash was zero".to_string()));
                }
                let output: Vec<u8> = arr.to_vec();
                result.push(output);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lib_alloc_cache() {
        let flags = RandomXFlags::default();
        let key = "Key";
        let cache = RandomXCache::new(flags, key.as_bytes()).expect("Failed to allocate cache");
        drop(cache);
    }

    #[test]
    fn lib_alloc_dataset() {
        let flags = RandomXFlags::default();
        let key = "Key";
        let cache = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let dataset =
            RandomXDataset::new(flags, cache.clone(), 0).expect("Failed to allocate dataset");
        drop(dataset);
        drop(cache);
    }

    #[test]
    fn lib_alloc_vm() {
        let flags = RandomXFlags::default();
        let key = "Key";
        let cache = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let mut vm =
            RandomXVM::new(flags, Some(cache.clone()), None).expect("Failed to allocate VM");
        drop(vm);
        let dataset = RandomXDataset::new(flags, cache.clone(), 0).unwrap();
        vm = RandomXVM::new(flags, Some(cache.clone()), Some(dataset.clone()))
            .expect("Failed to allocate VM");
        drop(dataset);
        drop(cache);
        drop(vm);
    }

    #[test]
    fn lib_dataset_memory() {
        let flags = RandomXFlags::default();
        let key = "Key";
        let cache = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let dataset = RandomXDataset::new(flags, cache.clone(), 0).unwrap();
        let memory = dataset.get_data().unwrap_or_else(|_| Vec::new());
        assert!(!memory.is_empty(), "Failed to get dataset memory");
        let v = vec![0u8; memory.len()];
        assert_ne!(memory, v);
        drop(dataset);
        drop(cache);
    }

    #[test]
    fn lib_calculate_hash() {
        let flags = RandomXFlags::get_recommended_flags();
        let flags2 = flags | RandomXFlags::FULLMEM;
        let key = "Key";
        let input = "Input";

        let cache1 = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let mut vm1 = RandomXVM::new(flags, Some(cache1.clone()), None).unwrap();
        let hash1 = vm1.calculate_hash(input.as_bytes()).expect("no data");
        let v = vec![0u8; hash1.len()];
        assert_ne!(hash1, v);
        assert!(vm1.reinit_cache(cache1.clone()).is_ok());
        let hash2 = vm1.calculate_hash(input.as_bytes()).expect("no data");
        assert_ne!(hash2, v);
        assert_eq!(hash1, hash2);

        let cache2 = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let vm2 = RandomXVM::new(flags, Some(cache2.clone()), None).unwrap();
        let hash3 = vm2.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(hash2, hash3);

        let cache3 = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let dataset3 = RandomXDataset::new(flags, cache3.clone(), 0).unwrap();
        let mut vm3 = RandomXVM::new(flags2, None, Some(dataset3.clone())).unwrap();
        let hash4 = vm3.calculate_hash(input.as_bytes()).expect("no data");
        assert_ne!(hash3, v);
        assert!(vm3.reinit_dataset(dataset3.clone()).is_ok());
        let hash5 = vm3.calculate_hash(input.as_bytes()).expect("no data");
        assert_ne!(hash4, v);
        assert_eq!(hash4, hash5);

        let cache4 = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let dataset4 = RandomXDataset::new(flags, cache4.clone(), 0).unwrap();
        let vm4 = RandomXVM::new(flags2, Some(cache4), Some(dataset4.clone())).unwrap();
        let hash6 = vm3.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(hash5, hash6);

        drop(dataset3);
        drop(dataset4);
        drop(cache1);
        drop(cache2);
        drop(cache3);
        drop(vm1);
        drop(vm2);
        drop(vm3);
        drop(vm4);
    }

    #[test]
    fn lib_calculate_hash_set() {
        let flags = RandomXFlags::default();
        let key = "Key";
        let inputs = vec![
            "Input".as_bytes(),
            "Input 2".as_bytes(),
            "Inputs 3".as_bytes(),
        ];
        let cache = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let vm = RandomXVM::new(flags, Some(cache.clone()), None).unwrap();
        let hashes = vm.calculate_hash_set(inputs.as_slice()).expect("no data");
        assert_eq!(inputs.len(), hashes.len());
        let mut prev_hash = Vec::new();
        for (i, hash) in hashes.into_iter().enumerate() {
            let v = vec![0u8; hash.len()];
            assert_ne!(hash, v);
            assert_ne!(hash, prev_hash);
            let compare = vm.calculate_hash(inputs[i]).unwrap(); // sanity check
            assert_eq!(hash, compare);
            prev_hash = hash;
        }
        drop(cache);
        drop(vm);
    }

    #[test]
    fn lib_calculate_hash_is_consistent() {
        let flags = RandomXFlags::get_recommended_flags();
        let key = "Key";
        let input = "Input";
        let cache = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let dataset = RandomXDataset::new(flags, cache.clone(), 0).unwrap();
        let vm = RandomXVM::new(flags, Some(cache.clone()), Some(dataset.clone())).unwrap();
        let hash = vm.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(
            hash,
            [
                114, 81, 192, 5, 165, 242, 107, 100, 184, 77, 37, 129, 52, 203, 217, 227, 65, 83,
                215, 213, 59, 71, 32, 172, 253, 155, 204, 111, 183, 213, 157, 155
            ]
        );
        drop(vm);
        drop(dataset);
        drop(cache);

        let cache1 = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let dataset1 = RandomXDataset::new(flags, cache1.clone(), 0).unwrap();
        let vm1 = RandomXVM::new(flags, Some(cache1.clone()), Some(dataset1.clone())).unwrap();
        let hash1 = vm1.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(
            hash1,
            [
                114, 81, 192, 5, 165, 242, 107, 100, 184, 77, 37, 129, 52, 203, 217, 227, 65, 83,
                215, 213, 59, 71, 32, 172, 253, 155, 204, 111, 183, 213, 157, 155
            ]
        );
        drop(vm1);
        drop(dataset1);
        drop(cache1);
    }

    #[test]
    fn lib_check_cache_and_dataset_lifetimes() {
        let flags = RandomXFlags::get_recommended_flags();
        let key = "Key";
        let input = "Input";
        let cache = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let dataset = RandomXDataset::new(flags, cache.clone(), 0).unwrap();
        let vm = RandomXVM::new(flags, Some(cache.clone()), Some(dataset.clone())).unwrap();
        drop(dataset);
        drop(cache);
        let hash = vm.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(
            hash,
            [
                114, 81, 192, 5, 165, 242, 107, 100, 184, 77, 37, 129, 52, 203, 217, 227, 65, 83,
                215, 213, 59, 71, 32, 172, 253, 155, 204, 111, 183, 213, 157, 155
            ]
        );
        drop(vm);

        let cache1 = RandomXCache::new(flags, key.as_bytes()).unwrap();
        let dataset1 = RandomXDataset::new(flags, cache1.clone(), 0).unwrap();
        let vm1 = RandomXVM::new(flags, Some(cache1.clone()), Some(dataset1.clone())).unwrap();
        drop(dataset1);
        drop(cache1);
        let hash1 = vm1.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(
            hash1,
            [
                114, 81, 192, 5, 165, 242, 107, 100, 184, 77, 37, 129, 52, 203, 217, 227, 65, 83,
                215, 213, 59, 71, 32, 172, 253, 155, 204, 111, 183, 213, 157, 155
            ]
        );
        drop(vm1);
    }

    #[test]
    fn randomx_hash_fast_vs_light() {
        let input = b"input";
        let key = b"key";

        let flags = RandomXFlags::get_recommended_flags() | RandomXFlags::FULLMEM;
        let cache = RandomXCache::new(flags, key).unwrap();
        let dataset = RandomXDataset::new(flags, cache, 0).unwrap();
        let fast_vm = RandomXVM::new(flags, None, Some(dataset)).unwrap();

        let flags = RandomXFlags::get_recommended_flags();
        let cache = RandomXCache::new(flags, key).unwrap();
        let light_vm = RandomXVM::new(flags, Some(cache), None).unwrap();

        let fast = fast_vm.calculate_hash(input).unwrap();
        let light = light_vm.calculate_hash(input).unwrap();
        assert_eq!(fast, light);
    }

    #[test]
    fn test_vectors_fast_mode() {
        // https://github.com/tevador/RandomX/blob/040f4500a6e79d54d84a668013a94507045e786f/src/tests/tests.cpp#L963-L979
        let key = b"test key 000";
        let vectors = [
            (
                b"This is a test".as_slice(),
                "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f",
            ),
            (
                b"Lorem ipsum dolor sit amet".as_slice(),
                "300a0adb47603dedb42228ccb2b211104f4da45af709cd7547cd049e9489c969",
            ),
            (
                b"sed do eiusmod tempor incididunt ut labore et dolore magna aliqua".as_slice(),
                "c36d4ed4191e617309867ed66a443be4075014e2b061bcdaf9ce7b721d2b77a8",
            ),
        ];

        let flags = RandomXFlags::get_recommended_flags() | RandomXFlags::FULLMEM;
        let cache = RandomXCache::new(flags, key).unwrap();
        let dataset = RandomXDataset::new(flags, cache, 0).unwrap();
        let vm = RandomXVM::new(flags, None, Some(dataset)).unwrap();

        for (input, expected) in vectors {
            let hash = vm.calculate_hash(input).unwrap();
            assert_eq!(hex::decode(expected).unwrap(), hash);
        }
    }

    #[test]
    fn test_vectors_light_mode() {
        // https://github.com/tevador/RandomX/blob/040f4500a6e79d54d84a668013a94507045e786f/src/tests/tests.cpp#L963-L985
        let vectors = [
            (
                b"test key 000",
                b"This is a test".as_slice(),
                "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f",
            ),
            (
                b"test key 000",
                b"Lorem ipsum dolor sit amet".as_slice(),
                "300a0adb47603dedb42228ccb2b211104f4da45af709cd7547cd049e9489c969",
            ),
            (
                b"test key 000",
                b"sed do eiusmod tempor incididunt ut labore et dolore magna aliqua".as_slice(),
                "c36d4ed4191e617309867ed66a443be4075014e2b061bcdaf9ce7b721d2b77a8",
            ),
            (
                b"test key 001",
                b"sed do eiusmod tempor incididunt ut labore et dolore magna aliqua".as_slice(),
                "e9ff4503201c0c2cca26d285c93ae883f9b1d30c9eb240b820756f2d5a7905fc",
            ),
        ];

        let flags = RandomXFlags::get_recommended_flags();
        for (key, input, expected) in vectors {
            let cache = RandomXCache::new(flags, key).unwrap();
            let vm = RandomXVM::new(flags, Some(cache), None).unwrap();
            let hash = vm.calculate_hash(input).unwrap();
            assert_eq!(hex::decode(expected).unwrap(), hash);
        }
    }
}
