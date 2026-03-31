import urllib.request
try:
    req = urllib.request.Request("https://raw.githubusercontent.com/08arpit/multi-object-tracking-sports/main/output/annotated_output.mp4", method="HEAD")
    with urllib.request.urlopen(req) as response:
        print("Status:", response.status)
        print("Headers:", response.headers)
except Exception as e:
    print("Error:", e)
