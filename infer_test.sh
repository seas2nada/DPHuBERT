# for root_dir in "exp/w2v2large_960toTED_dp_l2preg" "exp/w2v2large_960toTED_dp_nopreg" "exp/w2v2large_pretrained";
#     do . ./infer_chime.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
# done

# for root_dir in "exp/OMP/wav2vec2_asr-large-ls960-OMP-sp0.2"; do
#     . ./infer_ted.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
#     . ./infer_l2arc.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
#     . ./infer_cv.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
#     . ./infer_chime.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
# done

# for root_dir in "exp/w2v2large_960tol2arc_dp_l2preg_max5000_sp0.20_th0.4"; do
#     . ./infer_chime.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
# done

for exp_dir in "exp/whisper_medium_es+fr+pt+de_regnone_4gpu_max20000_sp0.20"; do
    model_name=${exp_dir}/ckpts/pruned_hubert_base.pth

    # for language in "ru" "it"; do 
    # bash infer_whisper.sh --exp_dir ${exp_dir} --model_name ${model_name} --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "wer" --dataset "mozilla-foundation/common_voice_16_1"; done

    for language in "ko_kr" "ja_jp" "cmn_hans_cn"; do
    bash infer_whisper.sh --exp_dir ${exp_dir} --model_name ${model_name} --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "cer" --dataset "google/fleurs"; done

    # for language in "ru_ru" "it_it"; do 
    # bash infer_whisper.sh --exp_dir ${exp_dir} --model_name ${model_name} --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "wer" --dataset "google/fleurs"; done
done

# for exp_dir in "exp/whisper_medium"; do
#     # model_name=${exp_dir}/ckpts/pruned_hubert_base.pth
#     model_name=pretrained/whisper-medium.hf.pth
    
#     for language in "ru"; do 
#     bash infer_whisper.sh --exp_dir ${exp_dir} --model_name ${model_name} --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "wer" --dataset "mozilla-foundation/common_voice_16_1"; done

#     # for language in "ru_ru"; do 
#     # bash infer_whisper.sh --exp_dir ${exp_dir} --model_name ${model_name} --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "wer" --dataset "google/fleurs"; done
    
#     # for language in "ko_kr"; do 
#     # bash infer_whisper.sh --exp_dir ${exp_dir} --model_name ${model_name} --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "cer" --dataset "google/fleurs"; done
# done

# exp_dir=exp/whisper_medium_es+fr+pt+de+tr+ko_regl2_8gpu_max30000_sp0.20_thre0.5
# model_name=${exp_dir}/ckpts/pruned_hubert_base.pth

# exp_dir=exp/whisper_medium
# model_name=pretrained/whisper-medium.hf.pth

# for language in "es" "fr" "pt" "de" "tr" "ko" "en" "it"; do 
# bash infer_whisper.sh --exp_dir ${exp_dir} --model_name ${model_name} --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "wer" --dataset "mozilla-foundation/common_voice_16_1"; done

# for language in "ja" "zh-CN"; do 
# bash infer_whisper.sh --exp_dir ${exp_dir} --model_name ${model_name} --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "cer" --dataset "mozilla-foundation/common_voice_16_1"; done

# for language in "it_it"; do 
# bash infer_whisper.sh --exp_dir ${exp_dir} --model_name ${model_name} --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "wer" --dataset "google/fleurs"; done

# for language in "ja_jp" "cmn_hans_cn"; do 
# bash infer_whisper.sh --exp_dir ${exp_dir} --model_name ${model_name} --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "cer" --dataset "google/fleurs"; done