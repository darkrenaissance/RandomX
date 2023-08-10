use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap();
    let n_threads = std::thread::available_parallelism()
        .unwrap()
        .get()
        .to_string();

    let cargo_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let build_dir = &cargo_dir.join("build");
    std::fs::create_dir_all(build_dir).unwrap();
    env::set_current_dir(build_dir).unwrap();

    // Generate CMake cache files
    let b = Command::new("cmake")
        .arg("-DARCH=native")
        .arg("..")
        .output()
        .expect("Failed to generate Makefile with CMake");
    std::io::stdout().write_all(&b.stdout).unwrap();
    std::io::stderr().write_all(&b.stderr).unwrap();
    assert!(b.status.success());

    // Build the library
    let b = Command::new("cmake")
        .arg("--build")
        .arg(".")
        .arg("--config")
        .arg("Release")
        .arg("-j")
        .arg(n_threads)
        .output()
        .expect("Failed to build RandomX library with CMake");
    std::io::stdout().write_all(&b.stdout).unwrap();
    std::io::stderr().write_all(&b.stderr).unwrap();
    assert!(b.status.success());

    env::set_current_dir(cargo_dir).unwrap();

    // Tell cargo how to find the static library
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.to_string_lossy()
    );
    println!("cargo:rustc-link-lib=static=randomx");

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else {
        unimplemented!()
    }

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/randomx.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
