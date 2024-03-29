#!/bin/bash
#
# This script prepares train and validation datasets for cross validation ML.
# It creates takes a Shuffler database on input and splits it into
# a train.db and a validation.db databases in N different ways.
# The output is N folders, each with train.db and validation.db databases inside.
#
# First the database is split into N equal parts. Each of N splits takes one
# of those part as the validation.db and the rest are merged into the train.db.


# Example:
#   tools/make_cross_validation_splits \
#     --input_db testdata/cars/micro1_v5.db \
#     --output_dir /tmp/cross_validation \
#     --number 3


# Parse command line arguments.

PROGNAME=${0##*/}
usage()
{
  cat << EO
This script prepares train and test datasets for cross validation ML.

Usage:
  $PROGNAME
      --input_db <input_db>
      --output_dir <output_dir>
      --number <number>
      [--seed <seed>]
      [-h|--help]

Options:
  --input_db
      (required) Path to input Shuffler database.
  --output_dir
      (required) Cross validation directory. It will be created if does not exist.
  --number
      (required) The number of cross-validation splits.
  --seed
      RNG seed.
  -h|--help
      Print usage and exit.
EO
}

ARGUMENT_LIST=(
    "input_db"
    "output_dir"
    "number"
    "seed"
)

opts=$(getopt \
    --longoptions "help,""$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "h" \
    -- "$@"
)

# Defaults.
seed=0
shuffler_bin=python -m shuffler

eval set --$opts

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --input_db)
            input_db=$2
            shift 2
            ;;
        --output_dir)
            output_dir=$2
            shift 2
            ;;
        --number)
            number=$2
            shift 2
            ;;
        --seed)
            seed=$2
            shift 2
            ;;
        --) # No more arguments
            shift
            break
            ;;
        *)
            echo "Arg '$1' is not supported."
            exit 1
            ;;
    esac
done

if [ -z "$input_db" ]; then
  echo "Argument input_db is required."
  exit 1
fi
if [ -z "$output_dir" ]; then
  echo "Argument output_dir is required."
  exit 1
fi
if [ -z "$number" ]; then
  echo "Argument number is required."
  exit 1
fi

# The end of the parsing code.
################################################################################


mkdir -p ${output_dir}

seq_from_zero_to_n_minus_one=$(seq 0 $((${number} - 1)))

# Compute the fraction of each part in format %.3f.
# For example, for N=5, fraction_size = 1 / 5 = .200
fraction_size=$(echo "scale=3; 1/$number" | bc)
echo "Will use fraction_size=${fraction_size}"

echo "number ${number}"

# Make input arguments for Shuffler.
out_names=""
out_fractions=""
for split in ${seq_from_zero_to_n_minus_one}; do
  echo "split $split"
  out_names="${out_names} part${split}.db"
  out_fractions="${out_fractions} ${fraction_size}"
done
echo "Will use out_names='${out_names}'"
echo "Will use out_fractions='${out_fractions}'"

# Cut into N parts.
python3 -m shuffler -i "${input_db}" \
    splitDb \
    --out_dir "${output_dir}" \
    --randomly \
    --seed ${seed} \
    --out_names ${out_names} \
    --out_fractions ${out_fractions}

status=$?
if [ ${status} -ne 0 ]; then
  echo "Shuffler's splitDb failed."
  exit ${status}
fi


# Make N splits.
for split in ${seq_from_zero_to_n_minus_one}; do

  # Make validation.db out of one part.
  mkdir -p "${output_dir}/split${split}"
  cp "${output_dir}/part${split}.db" "${output_dir}/split${split}/validation.db"

  # Make repeated cmd argument for Shuffler.
  addDb=""
  for part in ${seq_from_zero_to_n_minus_one}; do
    if [[ ${part} -ne ${split} ]]; then
      echo "part split ${part} ${split}"
      addDb="${addDb} addDb --db_file ${output_dir}/part${part}.db"
      # Add separator "|" between parts (including the last one.)
      addDb="${addDb} |"
    fi
  done
  addDb=${addDb::-2}  # Cut the last " |".
  echo "Will use addDb: '${addDb}'"

  # Make train.db out of all other parts.
  echo "python3 -m shuffler -o ${output_dir}/split${split}/train.db ${addDb}"
  python3 -m shuffler -o ${output_dir}/split${split}/train.db ${addDb}

  status=$?
  if [ ${status} -ne 0 ]; then
    echo "Shuffler's splitDb failed."
    exit ${status}
  fi

done

echo "The content of '${output_dir}' now is:"
echo $(ls ${output_dir})
