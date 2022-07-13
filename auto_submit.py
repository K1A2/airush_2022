import subprocess
import time


def run(cmd):
    completed = subprocess.run(["powershell", "-Command", cmd], capture_output=True)
    return completed

if __name__ == '__main__':
    n = [202]
    l = [[0, 20]]
    for j in range(2):
        for i in range(l[j][0], l[j][1]):
            is_succ = 0
            command = f"nsml submit KR96310/airush2022-1-2a/{n[j]} {i} --esm 96310"
            while not is_succ:
                print('execute commands:', command)
                info = run(command)
                if info.returncode != 0:
                    print("An error occured\n%s", info.stderr.decode("utf-8"))
                else:
                    print("executed successfully!")
                    print(info.stdout.decode("utf-8"))
                    is_succ = 1
                if not is_succ:
                    time.sleep(60 * 5)