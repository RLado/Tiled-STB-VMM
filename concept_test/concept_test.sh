# Cut the frames
python video2frames.py -i 14_25Hz_200mV_2.mp4 -o tile1 -c 505 546 -d 128 128
python video2frames.py -i 14_25Hz_200mV_2.mp4 -o tile2 -c 505 644 -d 128 128
python video2frames.py -i 14_25Hz_200mV_2.mp4 -o ground_truth -c 505 546 -d 128 226

# Magnify the sequences
python STB-VMM/run.py -j 1 -b 1 --load_ckpt STB-VMM/ckpt/ckpt_e49.pth.tar --save_dir tile1_mag -m 20 --video_path ./tile1/frame --num_data 749
python STB-VMM/run.py -j 1 -b 1 --load_ckpt STB-VMM/ckpt/ckpt_e49.pth.tar --save_dir tile2_mag -m 20 --video_path ./tile2/frame --num_data 749

echo "Reshaping frames..."
num_jobs="\j"  # The prompt escape for number of jobs currently running
max_procs=120 # Change to increase/decrease the concurrent reshaping calculations
for i in ./ground_truth/*.png; do
    while (( ${num_jobs@P} > $max_procs )); do
        wait -n
    done
    python3 ./STB-VMM/utils/auto_pad.py -i "$i" -d 64 -o "$i" &
done
python STB-VMM/run.py -j 1 -b 1 --load_ckpt STB-VMM/ckpt/ckpt_e49.pth.tar --save_dir ground_truth_mag -m 20 --video_path ./ground_truth/frame --num_data 749

# Stitch together
mkdir stitched_mag
A=$(ls ./tile1_mag/*.png)
B=$(ls ./tile2_mag/*.png)
for i in {0..750}; do
    echo $i
    j=${A[i]}
    k=${B[i]}
    python img_stitcher.py -i "$j" "$k" -c 0 0 0 98 -d 128 128 128 128 -o ./stitched_mag/frame_$i.png
done

# Convert to video
ffmpeg -framerate 60 -pattern_type glob -i './stitched_mag/*.png' stitched_output.mp4
ffmpeg -framerate 60 -pattern_type glob -i './ground_truth_mag/*.png' ground_truth_output.mp4