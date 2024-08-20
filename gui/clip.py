from pytube import YouTube


def download_youtube_video(url, output_path='.'):
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # Get the video stream with 720p resolution
        stream = yt.streams.filter( file_extension='mp4').first()

        if stream:
            # Download the video
            print(f"Downloading: {yt.title}")
            stream.download(output_path=output_path)
            print("Download completed!")
        else:
            print("No 720p stream available for this video.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
video_url = input("Enter the YouTube video URL: ")
download_youtube_video(video_url)
