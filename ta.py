import os
import asyncio
from telethon import TelegramClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Cấu hình Telegram
api_id = 28566035
api_hash = '4b5b80861d467593be229506682e4172'
phone = '+84345899390'
target_group = -1002013603722  # ID nhóm
topic_id = 80387  # ✅ ID chủ đề cần tải ảnh

BATCH_SIZE = 30000
wet_drops_folder = 'nakara'
os.makedirs(wet_drops_folder, exist_ok=True)

# Load mô hình đã huấn luyện
model_path = 'wet_drops_classifier_mobilenetv2.h5'
model = load_model(model_path)

# File theo dõi tiến độ
progress_file = 'downloaded_ids.txt'
range_file = 'id_range.txt'

# Load ID ảnh đã xử lý
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        downloaded_ids = set(int(line.strip()) for line in f)
else:
    downloaded_ids = set()

# Lấy ID tin nhắn mới nhất trong chủ đề
async def get_latest_message_id(client):
    message = await client.get_messages(target_group, limit=1, reply_to=topic_id)
    return message[0].id if message else 0

# Hàm kiểm tra ảnh có đúng kiểu hay không
def is_wanted_image(img_path, threshold=0.5):
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    print(f"→ Dự đoán: {pred:.4f}")
    return pred > threshold

# Hàm chính
async def main():
    client = TelegramClient('session_name', api_id, api_hash)
    await client.start(phone)

    async with client:
        latest_id = await get_latest_message_id(client)
        print(f"ID mới nhất trong chủ đề: {latest_id}")

        # Đọc khoảng ID
        if os.path.exists(range_file):
            with open(range_file, 'r') as f:
                start_id, end_id = map(int, f.read().split(','))
        else:
            start_id = latest_id
            end_id = max(start_id - BATCH_SIZE, 0)

        # Duyệt ảnh trong chủ đề
        async for message in client.iter_messages(
            target_group,
            min_id=end_id,
            max_id=start_id,
            reverse=True,
            reply_to=topic_id  # ✅ Chỉ lọc trong chủ đề này
        ):
            if message.photo and message.id not in downloaded_ids:
                tmp_path = await message.download_media(file='temp.jpg')
                if is_wanted_image(tmp_path):
                    save_path = os.path.join(wet_drops_folder, f'{message.id}.jpg')
                    os.rename(tmp_path, save_path)
                    print(f'✅ Ảnh ID {message.id} phù hợp → lưu tại {save_path}')
                    with open(progress_file, 'a') as f:
                        f.write(f'{message.id}\n')
                    downloaded_ids.add(message.id)
                else:
                    os.remove(tmp_path)
                    print(f'❌ Ảnh ID {message.id} không đúng kiểu → bỏ qua')

        # Cập nhật ID
        new_start = end_id - 1
        new_end = max(end_id - BATCH_SIZE, 0)
        with open(range_file, 'w') as f:
            f.write(f'{new_start},{new_end}')
        print(f"Lần sau sẽ xử lý từ ID {new_start} đến {new_end}")

if __name__ == '__main__':
    asyncio.run(main())
