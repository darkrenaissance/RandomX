#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

//! (Most of the code is taken from https://github.com/moneromint/randomx4r)
//! Rust bindings to librandomx, a library for computing RandomX hashes.
//!
//! # Examples
//!
//! ## Light mode hash
//!
//! Requires 256M of shared memory.
//!
//! ```no_run
//! use randomx::{RandomXCache, RandomXError, RandomXFlags, RandomXVM};
//!
//! // Get flags supported by this system.
//! let flags = RandomXFlags::default();
//! let cache = RandomXCache::new(flags, b"key")?;
//! let vm = RandomXVM::new(flags, &cache)?;
//! let hash = vm.hash(b"input"); // is a [u8; 32]
//! # Ok::<(), RandomXError>(())
//! ```
//!
//! ## Fast mode hash
//!
//! Requires 2080M of shared memory.
//!
//! ```no_run
//! use randomx::{RandomXDataset, RandomXError, RandomXFlags, RandomXVM};
//!
//! // OR the default flags with FULLMEM (aka. fast mode)
//! let flags = RandomXFlags::default() | RandomXFlags::FULLMEM;
//! // Speed up dataset initialisation
//! let threads = std::thread::available_parallelism().unwrap().get();
//! let dataset = RandomXDataset::new(flags, b"key", threads)?;
//! let vm = RandomXVM::new_fast(flags, &dataset)?;
//! let hash = vm.hash(b"input");
//! # Ok::<(), RandomXError>(())
//! ```
//!
//! # Errors
//!
//! Some operations (e.g. allocating a VM or dataset) can fail if the
//! system doesn't have enough free memory, or if you tried to force a
//! feature like large pages or AVX2 on a system that does not support
//! it.

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::marker::PhantomData;
use std::sync::Arc;
use std::thread;

use bitflags::bitflags;

#[derive(Debug, Copy, Clone)]
pub enum RandomXError {
    /// Occurs when allocating the RandomX cache fails.
    ///
    /// Reasons include:
    /// * Memory allocation fails
    /// * The JIT flag is set but the current platform does not support it
    /// * An invalid or unsupported ARGON2 value is set
    CacheAllocError,

    /// Occurs when allocating a RandomX dataset fails.
    ///
    /// Reasons include:
    /// * Memory allocation fails
    DatasetAllocError,

    /// Occurs when creating a VM fails.
    ///
    /// Reasons included:
    /// * Scratchpad memory allocation fails
    /// * Unsupported flags
    VmAllocError,
}

impl std::error::Error for RandomXError {}
impl std::fmt::Display for RandomXError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::CacheAllocError => write!(f, "Failed to allocate RandomX cache"),
            Self::DatasetAllocError => write!(f, "Failed to allocate RandomX dataset"),
            Self::VmAllocError => write!(f, "Failed to allocate RandomX VM"),
        }
    }
}

bitflags! {
    /// Represents options that can be used when allocating the
    /// RandomX dataset or VM.
    #[derive(Copy, Clone)]
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

impl Default for RandomXFlags {
    /// Get the recommended flags to use on the current machine.
    ///
    /// Does not include any of the following flags:
    /// * LARGEPAGES
    /// * JIT
    /// * SECURE
    fn default() -> Self {
        // Explode if bits do not match up
        unsafe { Self::from_bits(randomx_get_flags()).unwrap() }
    }
}

/// Dataset cache for light-mode hashing.
pub struct RandomXCache {
    pub(crate) cache: *mut randomx_cache,
}

impl RandomXCache {
    pub fn new(flags: RandomXFlags, key: &[u8]) -> Result<Self, RandomXError> {
        let cache = unsafe { randomx_alloc_cache(flags.bits()) };

        if cache.is_null() {
            return Err(RandomXError::CacheAllocError);
        }

        unsafe {
            randomx_init_cache(cache, key.as_ptr() as *const std::ffi::c_void, key.len());
        }

        Ok(RandomXCache { cache })
    }
}

impl Drop for RandomXCache {
    fn drop(&mut self) {
        unsafe { randomx_release_cache(self.cache) }
    }
}

unsafe impl Send for RandomXCache {}
unsafe impl Sync for RandomXCache {}

pub struct RandomXDataset {
    pub(crate) dataset: *mut randomx_dataset,
}

impl RandomXDataset {
    pub fn new(flags: RandomXFlags, key: &[u8], n_threads: usize) -> Result<Self, RandomXError> {
        assert!(n_threads > 0);

        let cache = RandomXCache::new(flags, key)?;
        let dataset = unsafe { randomx_alloc_dataset(flags.bits()) };

        if dataset.is_null() {
            return Err(RandomXError::DatasetAllocError);
        }

        let mut dataset = RandomXDataset { dataset };

        let count = unsafe { randomx_dataset_item_count() };

        if n_threads == 1 {
            unsafe {
                randomx_init_dataset(dataset.dataset, cache.cache, 0, count);
            }
        } else {
            let mut handles = Vec::new();
            let cache_arc = Arc::new(cache);
            let dataset_arc = Arc::new(dataset);

            let size = count / n_threads as u64;
            let last = count % n_threads as u64;
            let mut start = 0;

            for i in 0..n_threads {
                let cache = cache_arc.clone();
                let dataset = dataset_arc.clone();
                let mut this_size = size;
                if i == n_threads - 1 {
                    this_size += last;
                }
                let this_start = start;

                handles.push(thread::spawn(move || unsafe {
                    randomx_init_dataset(dataset.dataset, cache.cache, this_start, this_size);
                }));

                start += this_size;
            }

            for handle in handles {
                let _ = handle.join();
            }

            dataset = match Arc::try_unwrap(dataset_arc) {
                Ok(dataset) => dataset,
                Err(_) => return Err(RandomXError::DatasetAllocError),
            };
        }

        Ok(dataset)
    }
}

impl Drop for RandomXDataset {
    fn drop(&mut self) {
        unsafe { randomx_release_dataset(self.dataset) }
    }
}

unsafe impl Send for RandomXDataset {}
unsafe impl Sync for RandomXDataset {}

pub struct RandomXVM<'a, T: 'a> {
    vm: *mut randomx_vm,
    phantom: PhantomData<&'a T>,
}

impl RandomXVM<'_, RandomXCache> {
    pub fn new(flags: RandomXFlags, cache: &'_ RandomXCache) -> Result<Self, RandomXError> {
        if flags.contains(RandomXFlags::FULLMEM) {
            return Err(RandomXError::VmAllocError);
        }

        let vm = unsafe { randomx_create_vm(flags.bits(), cache.cache, std::ptr::null_mut()) };

        if vm.is_null() {
            return Err(RandomXError::VmAllocError);
        }

        Ok(Self {
            vm,
            phantom: PhantomData,
        })
    }
}

impl RandomXVM<'_, RandomXDataset> {
    pub fn new_fast(
        flags: RandomXFlags,
        dataset: &'_ RandomXDataset,
    ) -> Result<Self, RandomXError> {
        if !flags.contains(RandomXFlags::FULLMEM) {
            return Err(RandomXError::VmAllocError);
        }

        let vm = unsafe { randomx_create_vm(flags.bits(), std::ptr::null_mut(), dataset.dataset) };

        if vm.is_null() {
            return Err(RandomXError::VmAllocError);
        }

        Ok(Self {
            vm,
            phantom: PhantomData,
        })
    }
}

impl<T> RandomXVM<'_, T> {
    /// Calculate the RandomX hash of some data.
    ///
    /// ```no_run
    /// # // ^ no_run, this is already tested in the actual tests
    /// use randomx::*;
    /// let flags = RandomXFlags::default();
    /// let cache = RandomXCache::new(flags, "key".as_bytes())?;
    /// let vm = RandomXVM::new(flags, &cache)?;
    /// let hash = vm.hash("input".as_bytes());
    /// # Ok::<(), RandomXError>(())
    /// ```
    pub fn hash(&self, input: &[u8]) -> [u8; RANDOMX_HASH_SIZE as usize] {
        let mut hash = std::mem::MaybeUninit::<[u8; RANDOMX_HASH_SIZE as usize]>::uninit();

        unsafe {
            randomx_calculate_hash(
                self.vm,
                input.as_ptr() as *const std::ffi::c_void,
                input.len(),
                hash.as_mut_ptr() as *mut std::ffi::c_void,
            );

            hash.assume_init()
        }
    }
}

impl<T> Drop for RandomXVM<'_, T> {
    fn drop(&mut self) {
        unsafe { randomx_destroy_vm(self.vm) }
    }
}

unsafe impl<T> Send for RandomXVM<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_calc_hash() {
        let flags = RandomXFlags::default();
        let cache = RandomXCache::new(flags, "RandomX example key\0".as_bytes()).unwrap();
        let vm = RandomXVM::new(flags, &cache).unwrap();
        let hash = vm.hash("RandomX example input\0".as_bytes());
        let expected = [
            138, 72, 229, 249, 219, 69, 171, 121, 217, 8, 5, 116, 196, 216, 25, 84, 254, 106, 198,
            56, 66, 33, 74, 255, 115, 194, 68, 178, 99, 48, 183, 201,
        ];

        assert_eq!(expected, hash);
    }

    #[test]
    fn can_calc_hash_fast() {
        let flags = RandomXFlags::default() | RandomXFlags::FULLMEM;
        let n = thread::available_parallelism().unwrap().get();
        let dataset = RandomXDataset::new(flags, "RandomX example key\0".as_bytes(), n).unwrap();
        let vm = RandomXVM::new_fast(flags, &dataset).unwrap();
        let hash = vm.hash("RandomX example input\0".as_bytes());
        let expected = [
            138, 72, 229, 249, 219, 69, 171, 121, 217, 8, 5, 116, 196, 216, 25, 84, 254, 106, 198,
            56, 66, 33, 74, 255, 115, 194, 68, 178, 99, 48, 183, 201,
        ];

        assert_eq!(expected, hash);
    }
}
