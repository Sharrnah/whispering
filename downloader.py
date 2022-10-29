import urllib.request as urllib
import zipfile
import progressbar as pb


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = pb.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def download_extract(url, extract_dir):
    filehandle, _ = urllib.urlretrieve(url, None, MyProgressBar())
    with zipfile.ZipFile(filehandle, "r") as f:
        f.extractall(extract_dir)
