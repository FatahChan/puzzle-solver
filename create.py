import asyncio
import os
import random as r
import shutil
from pathlib import Path

async def main():
    tasks = []
    images_dir = "images"
    images_list = os.listdir(images_dir)
    num_cores = os.cpu_count()
    semaphore = asyncio.Semaphore(num_cores)
    
    for image_name in images_list:
        tasks.append(asyncio.create_task(create(image_name.replace(".jpg", ""), semaphore)))
    
    await asyncio.gather(*tasks)

async def create(id, semaphore, tries=0):
    async with semaphore:
        print(f"[{id}] Creating puzzle")
        random_num = r.randint(10, 44)
        path = os.path.join("puzzle", str(id))

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        current_path = Path().absolute()
        puzzle_path = os.path.join(current_path, "puzzle")
        image_path = os.path.join(current_path, "images")
        process = await asyncio.create_subprocess_exec(
            'piecemaker',
            f'--number-of-pieces={random_num}',
            '-d',
            f'{puzzle_path}/{id}',
            f'{image_path}/{id}.jpg',
            stdout=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == -9:
            shutil.rmtree(path)
            print(f'[{id}] failed try {tries}')
            if tries > 3:
                return
            await create(id, semaphore, tries + 1)
        else:
            print(f'[{id}] {stdout.decode()}')
            print(f"[{id}] Puzzle created")

# Run the main function
asyncio.run(main())
