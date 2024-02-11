# Xử lý ảnh với OpenCV

Chào mừng các bạn đến với phần Xử lý ảnh với OpenCV. Trong bài học này, chúng ta sẽ bắt đầu xây dựng kiến thức về cách sử dụng thư viện OpenCV, cụ thể là làm thế nào để mở ảnh và vẽ trên ảnh. Sau đó, chúng ta sẽ mở rộng, tìm hiểu thêm nhiều chức năng khác mà thư viện OpenCV có.

OpenCV (Open Source Computer Vision) là một thư viện chứa các hàm lập trình, chủ yếu về xử lý thị giác máy tính. Thư viện OpenCV được Intel tạo ra năm 1999 và nó được viết bằng C++. Trong môn học này, chúng ta sẽ sử dụng các python binding, nhờ đó, chúng ta có thể sử dụng trực tiếp các ngôn ngữ lập trình Python cũng như các thư viện Python như matplotlib, hoặc NumPy cùng với OpenCV.

Thị giác máy tính sử dụng máy tính để phân tích hình ảnh và video, tương tự với cách con người phân tích thứ gì đó mà họ đang xem, chẳng hạn như hình ảnh hoặc video, xác định khuôn mặt mà họ thấy trong hình ảnh. 

- ***\*Video: [Giới thiệu về OpenCV và Thị giác Máy tính](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12291386#overview)\****

### 1. Đọc/ghi ảnh

***\*1. Đọc/ghi ảnh\****

Trước tiên, chúng ta sẽ thảo luận chi tiết hơn về cách mở file ảnh trên notebook với OpenCV. Chúng ta sẽ sử dụng OpenCV kết hợp với matplotlib để trực tiếp đọc ảnh dưới dạng mảng và hiển thị nó.

- **Video: [Mở file hình ảnh trên notebook](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12257620#overview)**

Trong video này, chúng ta sẽ sử dụng thư viện OpenCV để hiển thị hình ảnh trên các cửa sổ riêng biệt bên ngoài Jupyter. Chúng ta sẽ cần hiển thị bên ngoài Jupyter với các phân tích video và hình ảnh phức tạp hơn mà chúng ta sẽ thực hiện sau này.

- **Video: [Mở file hình ảnh với OpenCV](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12257622#overview)**

### 2. Các thao tác trên ảnh

**2. Các thao tác trên ảnh**

Ở phần này, chúng ta sẽ tìm hiểu cách vẽ trên hình ảnh. Chúng ta sẽ bắt đầu với một vài hình dạng, chẳng hạn như hình chữ nhật và hình tròn, sau đó chuyển sang những thứ như vẽ văn bản lên hình ảnh.

- **Video: [Vẽ trên ảnh - Phần 1](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12257624#overview)**
- **Video: [Vẽ trên ảnh - Phần 2](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12291690#overview)**

Cho đến nay, chúng ta mới chỉ làm việc với không gian màu RGB, tức là mã hóa RGB hay mã hóa Red, Green, Blue. Các màu được mô hình hóa thành tổ hợp của Red, Green và Blue. Tuy nhiên, mã hóa RGB thực ra là một mã màu khá cũ. Trong video này, chúng ta sẽ tìm hiểu thêm về HSL, HSV và một biến thể khác của RGB - BGR.

- **Video: [Ánh xạ màu](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12257650#overview)**

Chúng ta thường sẽ xử lý nhiều hình ảnh và OpenCV có nhiều phương pháp lập trình trộn các hình ảnh với nhau và dán các hình ảnh lên trên nhau.

- **Video: [Trộn và dán hình ảnh](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12257654#overview)** 
- **Video tham khảo: [Trộn và dán hình ảnh - Phần 2](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12313088#overview)**

### 3. Tiền xử lý ảnh

 **3. Tiền xử lý ảnh**

Ở một số ứng dụng thị giác máy tính, ảnh màu thường được chuyển sang grayscale (thang độ xám), vì cuối cùng chỉ có các cạnh và shape là quan trọng đối với các ứng dụng nhất định. Tương tự, một số ứng dụng chỉ yêu cầu hình ảnh nhị phân hiển thị hình dạng khái quát, không chỉ grayscale, chỉ đen hoặc trắng. 

Nhìn chung, ngưỡng là một phương pháp rất đơn giản để phân đoạn hình ảnh thành các phần khác nhau. Ngưỡng nhị phân sẽ chuyển đổi một hình ảnh thành ảnh mới chỉ gồm hai giá trị: trắng hoặc đen. Sau đó, chúng ta sẽ thấy một số ví dụ khác về ngưỡng chuyển đổi thành một số loại grayscale - có các giá trị ở giữa màu đen và trắng.

- **Video: [Ngưỡng màu](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12257660#overview)**

Trong phần cuối của bài học, chúng ta sẽ tìm hiểu về làm mờ (blurring) và làm mịn (smoothing) ảnh. Làm mờ và làm mịn thường được kết hợp với phát hiện cạnh (edge detection).

- **Video: [Làm mờ và làm mịn ảnh](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12257660#overview)**
- **Video tham khảo: [Làm mờ và làm mịn ảnh - Phần 2](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12315302#overview)**

### Lab 2.1 - Xử lý ảnh với OpenCV

Chạy, đọc, làm theo các hướng dẫn trong notebook và điền vào tất cả các khối code trống cần thiết. Đảm bảo hoàn thành mọi thứ và trả lời tất cả các câu hỏi thiết yếu cho các lab sau:

- Lab 2.1.1: **Làm việc với ảnh**
- Lab 2.1.2: **Xử lý ảnh**

Các bạn download dữ liệu và notebook về tại [đây](https://drive.google.com/drive/folders/1rI4ZwgjePucJHWSQynSI6KOZAkvwlMTx?usp=sharing).

Nếu các bạn không hiểu được các đoạn code trong lab hay không biết giải quyết các vấn đề trong bài lab ra sao, có thể tham khảo phần lý giải chi tiết trong các video dưới đây:



- ***\*[Lab 2.1.1](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12257672#questions)\****
- ***\*[Lab 2.1.2](https://funix.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/learn/lecture/12257674#questions)\****



### Lab 2.2 - Hồi quy với Tensorflow

Trong notebook này, chúng ta sẽ đặt nền tảng cho cách lấy mẫu đầu vào (dữ liệu của bạn), xây dựng mạng nơ-ron để khám phá các mẫu (pattern) trong các đầu vào đó rồi đưa ra dự đoán (ở dạng số) dựa trên các đầu vào đó.

- **Lab 2.2:** [Hồi quy với Tensorflow](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/01_neural_network_regression_in_tensorflow.ipynb)