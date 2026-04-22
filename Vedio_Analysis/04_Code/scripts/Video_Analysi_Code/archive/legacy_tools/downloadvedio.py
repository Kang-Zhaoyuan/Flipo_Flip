import yt_dlp


def download_youtube_video(url):
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": "%(title)s.%(ext)s",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"正在准备下载: {url}")
        ydl.download([url])


if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=0VQAyx1AZBM"
    download_youtube_video(video_url)
