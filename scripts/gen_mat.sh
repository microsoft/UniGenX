#Please fill in your data path in the vacant code line below
CKPT=
CKPT_NAME=$(basename $CKPT)
CKPT_FOLDER=$(dirname $CKPT)

INPUT=

INPUT_FNAME=$(basename $INPUT)
OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl

if [ -f ${OUTPUT} ]; then
rm ${OUTPUT}
fi
if [ -f ${OUTPUT} ]; then
    echo "Output file ${OUTPUT} already exists. Skipping."
else
    python unigenx_infer.py \
    --dict_path unigenx/data/dict_mat.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer num \
    --infer --infer_batch_size 256 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \
    --verbose \
    --no_space_group \
    --target material \
    --diff_steps 200
fi

