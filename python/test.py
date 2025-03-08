import os
import shutil

def process_files(source_folder, target_folder):
    """
    从 source_folder 中读取文件，根据条件忽略某些文件，并将其他文件复制到 target_folder。
    条件：如果两个连续文件的文件名相同，且第一个文件的扩展名为.HEIC，第二个文件的扩展名为.MOV，则忽略这两个文件。
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取源文件夹中的所有文件
    files = sorted(os.listdir(source_folder))

    # 用于标记是否忽略当前文件
    ignore_next = False

    for i in range(len(files)):
        file = files[i]
        file_path = os.path.join(source_folder, file)
        
        # 如果当前文件需要被忽略
        if ignore_next:
            ignore_next = False
            continue

        # 检查是否是最后一个文件
        if i == len(files) - 1:
            # 如果是最后一个文件，直接复制
            shutil.copy(file_path, target_folder)
            continue

        # 获取当前文件和下一个文件的名称和扩展名
        current_name, current_ext = os.path.splitext(file)
        next_file = files[i + 1]
        next_name, next_ext = os.path.splitext(next_file)

        # 检查是否满足忽略条件
        if (current_name == next_name and
            current_ext.lower() == '.heic' and
            next_ext.lower() == '.mov'):
            # 标记下一个文件需要被忽略
            ignore_next = True
            continue

        # 如果不满足忽略条件，复制当前文件
        shutil.copy(file_path, target_folder)

    print("处理完成！")

# 示例用法
source_folder = r"E:\新建文件夹\2022"
target_folder = r"E:\新建文件夹\2022未完成"
process_files(source_folder, target_folder)