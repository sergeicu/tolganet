function time_topup {
	echo "Begin ${hcp}"
	cd ".../${hcp}/"

	# Define the list of directories to process
	directories=("dir95" "dir96" "dir97")

	# Loop through each directory
	for dir in "${directories[@]}"; do
		hcp="${hcp}_${dir}" # Update hcp variable with the current directory

		# Extract b0 images
		fslroi "${hcp}_3T_DWI_${dir}_LR" "${hcp}_3T_DWI_${dir}_LR_b0" 0 1
		fslroi "${hcp}_3T_DWI_${dir}_RL" "${hcp}_3T_DWI_${dir}_RL_b0" 0 1

		# Merge b0 images
		fslmerge -t "${hcp}_3T_DWI_${dir}_RLRL_b0" "${hcp}_3T_DWI_${dir}_LR_b0" "${hcp}_3T_DWI_${dir}_RL_b0"
		echo "fslmerge"
		# Run TOPUP
		/usr/bin/time -p topup --imain="${hcp}_3T_DWI_${dir}_RLRL_b0" --datain=acqparams_DWI.txt --config=b02b0_1.cnf --out="${hcp}_3T_DWI_${dir}_topup_b0" --iout="${hcp}_3T_DWI_${dir}_topup_iout" --fout="${hcp}_3T_DWI_${dir}_topup_fout"

		# Apply distortion correction
		/usr/bin/time -p applytopup --imain="${hcp}_3T_DWI_${dir}_LR","${hcp}_3T_DWI_${dir}_RL" --inindex=1,2 --datain=acqparams_DWI.txt --topup="${hcp}_3T_DWI_${dir}_topup" --out="${hcp}_3T_DWI_${dir}_topup_results"

		# Output message indicating completion of processing for the current directory
		echo "Processing completed for ${dir}."
	done
}

subfolder_names=("150019" "199958" "299760" "996782")

for hcp in "${subfolder_names[@]}"; do
    echo
    echo ${hcp}
    echo
done