from models.classifier import predict

if __name__ == "__main__":
    image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRtpojAldVAnSDss070FIY5Api9zEh_r59h6g&s"  # Change this URL
    try:
        result = predict(image_url)
        print(result)
    except ValueError as e:
        print(e)
