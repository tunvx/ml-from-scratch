# SVM

Ý tưởng của SVM là tìm một siêu phẳng (hyper lane) để phân tách các điểm dữ liệu. Siêu phẳng này sẽ chia không gian thành các miền khác nhau và mỗi miền sẽ chứa một loại dữ liệu.

Vấn đề là có rất nhiều siêu phẳng, chúng ta phải chọn cái nào để tối ưu nhất? Siêu phẳng tối ưu mà chúng ta cần chọn là siêu phẳng phân tách có lề lớn nhất. Lý thuyết học máy đã chỉ ra rằng một siêu phẳng như vậy sẽ cực tiểu hóa giới hạn lỗi mắc phải.

- ***\*Video: [Thuật toán SVM](https://funix.udemy.com/course/machinelearning/learn/lecture/5714406#overview)\****

Chúng ta có thể tạo ranh giới quyết định phi tuyến rất phức tạp theo 2 cách: Ánh xạ dữ liệu đến các chiều không gian nhiều chiều hơn hoặc sử dụng các Kernel.

- ***\*Video: [Hạt nhân SVM (kernel)](https://funix.udemy.com/course/machinelearning/learn/lecture/6113144#overview)\****

Ánh xạ dữ liệu đến không gian nhiều chiều hơn sẽ giúp bạn đưa đường phi tuyến phức tạp trở thành 1 siêu phẳng tuyến tính trong chiều không gian mới. Tuy nhiên, cách làm này không hiệu quả lắm về mặt tài nguyên tính toán.

- ***\*Video: [Ánh xạ đến không gian nhiều chiều hơn](https://funix.udemy.com/course/machinelearning/learn/lecture/6113148#overview)\****

Sử dụng các hàm Kernel mô tả **quan hệ giữa hai điểm dữ liệu bất kỳ** trong không gian mới, thay vì đi tính toán trực tiếp **từng điểm dữ liệu trong không gian nhiều chiều mới** sẽ giúp bạn có thể tạo ranh giới quyết định phi tuyến rất phức tạp.

- ***\*Video: [Các thủ thuật Kernel](https://funix.udemy.com/course/machinelearning/learn/lecture/6113150#overview)\****

Ở video này, chúng ta sẽ được giới thiệu về một số loại Kernel khác nhau:

- - Gaussian RBF Kernel
  - Sigmoid Kernel
  - Polynomial Kernel

- ***\*Video: [Các dạng hàm Kernel](https://funix.udemy.com/course/machinelearning/learn/lecture/6113152#overview)\****

***\*Tài liệu tham khảo\****

Nếu bạn muốn tìm hiểu thêm về soft-margin trong SVM, có thể xem thêm ở [đây](https://www.coursera.org/learn/machine-learning/lecture/sHfVT/optimization-objective).

**Video: [(Optional) Kernel SVR không tuyến tính](https://funix.udemy.com/course/machinelearning/learn/lecture/19505940#overview)**



### Lab: Phân loại bình luận với SVM

- [Notebook](https://drive.google.com/drive/folders/1PgXpJASiilBH-SJC3qUwGJ7FOzhT9NoO?usp=sharing)
- [Important_words.json](https://drive.google.com/file/d/1a9_ey7pXNM9H-q7xFBce5oQ0m8hcCJML/view?usp=share_link)
- [amazon_baby_subset.csv](https://drive.google.com/file/d/1FJvQDNAL2lliJeUhCxM2-CZIf6HGZqD5/view?usp=share_link)

