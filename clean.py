import os
import shutil


def main():
    result_root_dir = 'out'
    cnt = 0
    for sub_dir in os.listdir(result_root_dir):
        sub_dir = os.path.join(result_root_dir, sub_dir)
        if os.path.isdir(sub_dir):
            if len(os.listdir(sub_dir)) != 10:
                print(sub_dir)
                cnt += 1
                shutil.rmtree(sub_dir)
    print(cnt)


if __name__ == '__main__':
    main()
