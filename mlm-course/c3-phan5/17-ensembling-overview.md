# Ensembling

**Ensembling** là các phương pháp kết hợp của nhiều mô hình học máy khác nhau để có được dự đoán tốt hơn về mặt chất lượng và nâng cao tính khái quát hóa (generalization) của mô hình.

Trong video này, chúng ta sẽ tìm hiểu về 3 phương pháp kết hợp cơ bản:

- - Phương pháp lấy trung bình (còn gọi là blending).
  - Phương pháp lấy trung bình có sử dụng trọng số: Mỗi mô hình khác nhau sẽ có 1 trọng số khác nhau, đơn giản nhất là các mô hình có kết quả ở validation data cao hơn sẽ có trọng số cao hơn.
  - Phương pháp lấy trung bình có điều kiện: Kết hợp các mô hình theo các điều kiện khác nhau tùy vào các tính chất đặc biệt của các đặc trưng.

- **Video: [Giới thiệu về các phương pháp Ensemble](https://drive.google.com/file/d/13XmXO-bXZu4aNRUC737oTgQWnt1QARrd/view)**

Bagging được đề xuất bởi Leo Breiman xây dựng một lượng lớn các model (thường là cùng loại) trên những subsamples khác nhau từ tập training dataset (random sample trong 1 dataset để tạo 1 dataset mới). Những model này sẽ được train độc lập và song song với nhau nhưng đầu ra của chúng sẽ được trung bình cộng để cho ra kết quả cuối cùng. Bagging cũng là ý tưởng chính để xây dựng nên một trong những thuật toán phân loại tốt nhất - Random Forest (rừng ngẫu nhiên) chính là sự kết hợp giữa thuật toán cây quyết định và bagging .

Phương pháp bagging được trình bày như sau:

- - Đầu tiên, cho một tập dữ liệu và một thuật toán đơn bất kỳ.
  - Trong tập dữ liệu ban đầu, phương pháp Bootstrap được sử dụng để chia tập dữ liệu ban đầu thành các tập dữ liệu huấn luyện con có kích thước bằng nhau.
  - Tiếp theo áp dụng một thuật toán đối với từng tập dữ liệu huấn luyện riêng tương ứng với một mô hình dự đoán. Cuối cùng kết quả dự đoán sẽ sử dụng giá trị trung bình "Mean" hoặc "Voting".

Một số tham số quan trọng trong phương pháp bagging:

- - Random seed khi trích chọn dữ liệu, khi shuffle dữ liệu trước khi đưa vào huấn luyện.
  - Số lượng đặc trưng trích chọn bạn sẽ sử dụng, cách thức trích chọn các đặc trưng này.
  - Các tham số cụ thể khác của mô hình mà bạn sử dụng để thực thi bagging.
  - Số lượng mô hình bạn sẽ sử dụng cho bagging.

- **Video: [Bagging](https://drive.google.com/file/d/13as-FB2yYUxMW6aFti62o1G4FWvosWRq/view)**

Boosting được đề xuất bởi Robert E Schapire xây dựng một lượng lớn các models (thường là cùng loại). Mỗi model sau sẽ học cách sửa những errors của model trước (dữ liệu mà model trước dự đoán sai) tạo thành một chuỗi các model mà model sau sẽ tốt hơn model trước bởi trọng số được update qua mỗi model (cụ thể ở đây là trọng số của những dữ liệu dự đoán đúng sẽ không đổi, còn trọng số của những dữ liệu dự đoán sai sẽ được tăng thêm). 

Hai kỹ thuật boosting được giới thiệu trong video này là weighted và residual. Weighted sử dụng cột dữ liệu mới được tạo ra từ errors của model trước như 1 đặc trưng mới cho các model khác. Adaboost là một trong những thuật toán điển hình áp dụng phương pháp này. Trong khi đó, Residual sử dụng dữ liệu errors này như một biến mục tiêu - một "y" mới cho model tiếp theo dự đoán. Xgboost và Lightgbm là 2 thuật toán tiêu biểu cho trường phái này. Trong những năm gần đây, Residual thường đem lại chất lượng tốt hơn so với Weighted.

Một số tham số quan trọng trong phương pháp boosting:

- - Tốc độ học (learning rate).
  - Số lượng các ước lượng (estimator).
  - Số lượng đặc trưng trích chọn bạn sẽ sử dụng, cách thức trích chọn các đặc trưng này.
  - Mô hình gốc để sử dụng cho boosting.
  - Kỹ thuật boosting bạn sử dụng

- **Video: [Boosting](https://drive.google.com/file/d/13fxjEoc5Dg97SdMy0udFCtcmqp-gvkR9/view)**

Stacking là một biến thể, còn được gọi là phương pháp meta-learning, bao gồm một hệ thống phân cấp các bộ phân loại khác nhau. Mục tiêu của stacking là để xây dựng một bộ phân loại cấp độ meta có thể dự đoán nhãn đích của tập dữ liệu bằng cách kết hợp kết quả các dự đoán từ các bộ phân loại riêng biệt. Tương tự như Boosting, Stacking sử dụng các sơ đồ trọng số phức tạp so với bagging sử dụng các sơ đồ trọng số đồng nhất đơn giản.

- **Video: [Stacking](https://drive.google.com/file/d/13fzHHMeKCqLelEMe1cDJFj0OoI9jXKIv/view)**

StackNet sử dụng các DNN làm bộ phân loại cấp độ meta hoặc có nhiều tầng, kết hợp nhiều biến thể đồ sộ và phức tạp hơn.

- **Video: [StackNet](https://drive.google.com/file/d/13scWphpVY2EAqU3Tj1gnrrJuEmHQhDui/view)**

Trong video này, tác giả tổng hợp lại các tips và tricks khi sử dụng Ensemble, gồm các phương pháp tác động lên mô hình (độ phức tạp, mức độ nông/sâu của mô hình), dữ liệu (trích chọn dữ liệu đưa vào huấn luyện, các phương pháp xử lý và trích xuất đặc trưng). 
Các bạn có thể tham khảo thêm ở [StackNet của H2O](https://github.com/h2oai/pystacknet) - thư viện mã nguồn mở về ensembling trong học máy.

- **Video: [Các tip và trick khi sử dụng Esemble](https://drive.google.com/file/d/141gy3c16t8Lu5bNj0D1-oUglV7JDZ3s9/view)**