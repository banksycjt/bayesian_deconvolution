import multiprocessing
import time

def worker(process_id):
    print(f"Process {process_id} started")
    time.sleep(2)  # 模拟耗时操作
    print(f"Process {process_id} finished")

if __name__ == "__main__":
    processes = []
    for i in range(5):
        # 创建进程
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("All processes completed")