#!/bin/bash

# ============================
# 配置部分
# ============================

# 远程服务器登录信息
REMOTE_USER="user_liz"
REMOTE_HOST1="192.168.110.110"
REMOTE_HOST2="192.168.110.227"
REMOTE_HOST3="192.168.110.228"
REMOTE_PORT="2222"

# 远程和本地基础路径（请确保最后没有斜杠，方便后续拼接路径）
REMOTE_BASE1="/home/user_liz/works_liz/isaac_works/rl_sim_env/logs/amp_vae/grq20_v1d6_amp_vae"
REMOTE_BASE2="/home/user_liz/works_liz/isaac_work/rl_sim_env/logs/amp_vae/grq20_v1d6_amp_vae"
REMOTE_BASE3="/home/user_liz/works_liz/isaac_work/rl_sim_env/logs/amp_vae/grq20_v1d6_amp_vae"
REMOTE_BASE4="/home/user_liz/works_liz/isaac_work/gpu2/isaac_works/rl_sim_env/logs/amp_vae/grq20_v1d6_amp_vae"
REMOTE_BASE5="/home/user_liz/works_liz/isaac_work/gpu2/rl_sim_env/logs/amp_vae/grq20_v1d6_amp_vae"
LOCAL_BASE="/home/lizhen/works/code/github/isaac_works/rl_sim_env/logs/amp_vae/grq20_v1d6_amp_vae"

# 远程文件夹列表（相对于 REMOTE_BASE 的子文件夹）
REMOTE_FOLDERS=("2025-06-24_01-00-19_s228_g4_cmd_test3"
                "2025-06-24_00-46-09_s227_g0_motor_random"
                "2025-06-24_00-38-41_s227_g4_cmd_test"
                "2025-06-23_23-39-23_s228_g0_cmd_test2"
                )

# 文件名列表（每个文件夹中要查找的文件）
FILE_LIST=("policy.yaml" "model_15000.pt")

# ============================
# 脚本逻辑部分
# ============================

# 在本地路径下创建对应的远程文件夹
for folder in "${REMOTE_FOLDERS[@]}"; do
    mkdir -p "$LOCAL_BASE/$folder"
    echo "已创建或存在本地文件夹: $LOCAL_BASE/$folder"
done

# 遍历每个远程文件夹和文件，检查远程文件是否存在，如果存在则复制到对应的本地文件夹
for folder in "${REMOTE_FOLDERS[@]}"; do
    for file in "${FILE_LIST[@]}"; do
        REMOTE_FILE1="$REMOTE_BASE1/$folder/$file"
        REMOTE_FILE2="$REMOTE_BASE2/$folder/$file"
        REMOTE_FILE3="$REMOTE_BASE3/$folder/$file"
        REMOTE_FILE4="$REMOTE_BASE4/$folder/$file"
        REMOTE_FILE5="$REMOTE_BASE5/$folder/$file"
        echo "复制远程文件: $REMOTE_FILE ..."

        # 通过 ssh 检查远程文件是否存在
        # if ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "[ -f '$REMOTE_FILE' ]"; then
        #     echo "发现文件 $file ，开始复制到 $LOCAL_BASE/$folder/"
        #     # 使用 scp 复制文件
            # scp -P "$REMOTE_PORT" ${REMOTE_USER}@${REMOTE_HOST}:"$REMOTE_FILE" "$LOCAL_BASE/$folder/"
            scp -P "$REMOTE_PORT" ${REMOTE_USER}@${REMOTE_HOST1}:"$REMOTE_FILE1" "$LOCAL_BASE/$folder/"
            scp -P "$REMOTE_PORT" ${REMOTE_USER}@${REMOTE_HOST2}:"$REMOTE_FILE2" "$LOCAL_BASE/$folder/"
            scp -P "$REMOTE_PORT" ${REMOTE_USER}@${REMOTE_HOST3}:"$REMOTE_FILE3" "$LOCAL_BASE/$folder/"
            scp -P "$REMOTE_PORT" ${REMOTE_USER}@${REMOTE_HOST2}:"$REMOTE_FILE4" "$LOCAL_BASE/$folder/"
            scp -P "$REMOTE_PORT" ${REMOTE_USER}@${REMOTE_HOST3}:"$REMOTE_FILE5" "$LOCAL_BASE/$folder/"
            # echo "在文件夹 $folder 中未找到文件 $file ，跳过。"

    done
done
