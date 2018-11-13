import requests


class Upload:
    def upload(self, modelPath, logPath, imgPath):
        return
        upload_url = ''
        files = {
            'model': open(modelPath, 'rb'),
            'log': open(logPath, 'rb'),
            'img': open(imgPath, 'rb')
        }
        r = requests.post(upload_url, files=files)
        print("status code: " + str(r.status_code))
        print(r.content)
