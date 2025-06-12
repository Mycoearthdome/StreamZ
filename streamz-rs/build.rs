use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("train_files.rs");
    let mut out_file = File::create(&dest_path).unwrap();
    let list_path = Path::new("train_files.txt");
    if list_path.exists() {
        let content = fs::read_to_string(list_path).unwrap();
        let mut entries = Vec::new();
        for line in content.lines() {
            let mut parts = line.split(',');
            if let (Some(path), Some(class)) = (parts.next(), parts.next()) {
                if let Ok(cls) = class.parse::<usize>() {
                    entries.push((path.trim().to_string(), cls));
                }
            }
        }
        writeln!(out_file, "const TRAIN_FILES: [(&str, usize); {}] = [", entries.len()).unwrap();
        for (p, c) in entries {
            writeln!(out_file, "    (\"{}\", {}),", p, c).unwrap();
        }
        writeln!(out_file, "];" ).unwrap();
    } else {
        writeln!(out_file, "const TRAIN_FILES: [(&str, usize); 6] = [").unwrap();
        writeln!(out_file, "    (\"examples/training_data/arctic_a0008.wav\", 0),").unwrap();
        writeln!(out_file, "    (\"examples/training_data/arctic_a0015.wav\", 0),").unwrap();
        writeln!(out_file, "    (\"examples/training_data/arctic_a0021.wav\", 0),").unwrap();
        writeln!(out_file, "    (\"examples/training_data/arctic_b0196.wav\", 1),").unwrap();
        writeln!(out_file, "    (\"examples/training_data/arctic_b0356.wav\", 1),").unwrap();
        writeln!(out_file, "    (\"examples/training_data/arctic_b0417.wav\", 1),").unwrap();
        writeln!(out_file, "];" ).unwrap();
    }
}
