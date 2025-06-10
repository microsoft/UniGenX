#Please fill in your data path in the vacant code line below
CKPT=

CKPT_FOLDER=$(dirname $CKPT)
CKPT_NAME=$(basename $CKPT)

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
    --dict_path unigenx/data/dict_drugs.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer num \
    --infer --infer_batch_size 16 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \
    --verbose \
    --diff_steps 200 \
    --target mol
fi
