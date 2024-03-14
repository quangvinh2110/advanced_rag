import aiohttp
import asyncio
import time
import json
import argparse

from tqdm.asyncio import tqdm

start_time = time.time()


def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--links_path",
        type=str,
        default=None,
        help="The absolute path to link file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The absolute path to output file"
    )
    args = parser.parse_args()

    # Sanity checks
    if args.links_path is None or args.output_path is None:
        raise ValueError("Need both a links file and a output file")
    
    return args


async def get_page_content(session, url):
    
    try:
        async with session.get(url) as resp:
            page_content = await resp.text()
            return (page_content, True)
    except Exception as err:
        return (f"{err}", False)


async def main(urls, output_path):

    async with aiohttp.ClientSession() as session:

        tasks = []
        for url in urls:
            tasks.append(asyncio.ensure_future(
                get_page_content(session, url)
            ))

        results = await tqdm.gather(*tasks)
        data = []
        failed_urls = []
        for url, (page_content, succeed) in zip(urls, results):
            if succeed:
                data.append(
                    {"url": url, "content": page_content}
                )
            else:
                failed_urls.append(url)
        with open(output_path, "a") as f:
            for sample in data: 
                d = json.dumps(sample, ensure_ascii=False) + "\n"
                f.write(d)
        failed_urls_path = "/".join(output_path.split("/")[:-1]) + "/failed_links.txt"
        with open(failed_urls_path, "a") as f:
            for url in failed_urls:
                # write each item on a new line
                f.write("%s\n" % url)


if __name__ == "__main__":
    args = parse_args()
    with open(args.links_path, "r") as f:
        urls = f.read().split("\n")
    asyncio.run(main(urls, args.output_path))
    print("--- %s seconds ---" % (time.time() - start_time))
