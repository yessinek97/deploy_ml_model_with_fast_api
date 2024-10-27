import requests
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--api-url",
    type=str,
    required=True
)

if __name__ == '__main__':
    args = parser.parse_args()
    api_url = args.api_url
    r = requests.post(api_url)
    
    print(f"request finished with code {r.status_code}")
    print(f"response content: {r.json()}")