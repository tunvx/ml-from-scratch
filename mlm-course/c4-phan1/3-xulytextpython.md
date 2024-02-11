# Xử lý text trong python

Chào mừng các bạn đến với phần xử lý text với Python. Mục tiêu của phần này là hiểu cách mở file .txt và file .pdf chuẩn chỉ với một vài thư viện Python cơ bản, gồm các chức năng được tích hợp sẵn và không có thư viện bên ngoài. Sau đó, chúng ta sẽ tìm hiểu một số biểu thức cơ bản thường gặp như tìm pattern trong text với regex.

- ***\*Video: [Giới thiệu về xử lý text với Python](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12744531#reviews)\****



### 1. Làm việc với text file

Trong video phần một này, chúng ta sẽ tập trung vào một số định dạng in cơ bản, cụ thể là f-string literal. Chúng ta cũng sẽ thảo luận về các tùy chọn căn chỉnh với các f-string literal.

- **Video: [Làm việc với text file trên Python - Phần 1](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12744537#reviews)**

Tiếp theo, chúng ta sẽ tìm hiểu sâu về làm việc với text file: tạo file, mở file, đọc và ghi file.

- **Video: [Làm việc với text file trên Python - Phần 2](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827455#reviews)**

Thường thì bạn có thể cần đọc dữ liệu văn bản từ tệp PDF thay vì tệp văn bản bình thường. Tuy rằng không phải tất cả các file văn bản ở dạng PDF đều có thể trích xuất, bạn có thể sử dụng thư viện PYPDF2 để làm việc này.

- **Video tham khảo: [Làm việc với tệp PDFs](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827469#reviews)**

### 2. Xử lý text nâng cao

Nếu bạn cần tìm một chuỗi thông tin dưới dạng một chuỗi text trong một tệp văn bản (lấy email/số điện thoại trong một thư điện tử, lấy thông tin điểm thi ở trong free form ...), regular expression là điều đầu tiên bạn nên nghĩ đến và thử nghiệm.

- **Video tham khảo: [Regular Expressions - Phần 1](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827479#reviews)**
- **Video tham khảo: [Regular Expressions Phần 2](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827483#reviews)** 

### Lab 3 - Phân loại với Tensorflow

Trong notebook này, chúng ta sẽ xem xét nhiều bài toán phân loại khác nhau với TensorFlow. Nói cách khác, chúng ta sẽ lấy một tập hợp đầu vào và dự đoán xem tập hợp đó thuộc lớp nào.

- **Lab 3:** **[Phân loại với Tensorflow](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/02_neural_network_classification_in_tensorflow.ipynb)**