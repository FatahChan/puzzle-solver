import os
import shutil
import json
import asyncio


async def cleanFiles(id):
    try:
        raster_dir = os.path.join("puzzle", str(id), "size-100", "raster")
        pieces_file = os.path.join(
            "puzzle", str(id), "size-100", "pieces.json")
        if (os.path.isdir(raster_dir)):

            print(f"[${id}] Cleaning files")
            shutil.move(raster_dir, os.path.join("puzzle", str(id), "raster"))
        if (os.path.isfile(pieces_file)):
            # read json file
            with open(pieces_file, 'r') as f:
                data = json.load(f)
            positions = []
            for i in data:
                positions.append(data[i][0])
                positions.append(data[i][1])
                positions.append(0)
            while len(positions) < 192:
                positions.append(-1)
            positions_obj = {
                "positions": positions
            }
            with open(os.path.join("puzzle", str(id), "positions.json"), 'w') as f:
                json.dump(positions_obj, f)
            if (len(positions) % 3 == 0):
                print(f"[${id}] Cleaning files")
            else:
                print(f"[${id}] Error cleaning files")
    except Exception as e:
        print(e)
        print(f"[${id}] Error cleaning files")
        pass


async def clean():
    tasks = []
    for i in os.listdir("images"):
        tasks.append(asyncio.create_task(cleanFiles(i.replace(".jpg", ""))))
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(clean())
