//! randomx example that calculates many hashes using multiple threads

use randomx::*;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use std::vec::Vec;

fn main() {
    const NUM_THREADS: u32 = 8;
    // number of hashes to perform in each thread, not the total.
    const NUM_HASHES: u32 = 5000;

    let start = Instant::now();

    // Try adding `| RandomXFlags::LARGEPAGES`.
    let flags = RandomXFlags::default() | RandomXFlags::FULLMEM;
    let dataset = Arc::new(RandomXDataset::new(flags, b"key", NUM_THREADS as usize).unwrap());

    println!("Dataset initialised in {}ms", start.elapsed().as_millis());

    let mut handles = Vec::new();

    let start = Instant::now();

    for i in 0..NUM_THREADS {
        let dataset = dataset.clone();

        handles.push(thread::spawn(move || {
            let mut nonce: u32 = i;
            let vm = RandomXVM::new_fast(flags, &dataset).unwrap();

            for _ in 0..NUM_HASHES {
                let _ = vm.hash(&nonce.to_be_bytes());

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
        "Completed {} hashes in {}ms",
        NUM_THREADS * NUM_HASHES,
        start.elapsed().as_millis()
    );
}
