# Hồi quy cơ bản



### 1. Mô hình hồi quy tuyến tính cơ bản, cách sử dụng và diễn giải

Hồi quy là một trong những công cụ thống kê và học máy quan trọng và được sử dụng rộng rãi nhất hiện nay. Nó cho phép bạn đưa ra dự đoán từ dữ liệu bằng cách tìm hiểu mối quan hệ giữa các đặc điểm của dữ liệu và một số phản hồi được quan sát, có giá trị liên tục. Hồi quy được sử dụng trong một số lượng lớn các ứng dụng khác nhau, từ dự đoán giá cổ phiếu đến tìm hiểu các mạng lưới điều hòa gen.

Khóa học của chúng ta bắt đầu với các khái niệm cơ bản nhất: Chỉ khớp đầu ra với một đặc trưng. Mô hình đơn giản này hình thành các dự đoán từ một đặc trưng đơn lẻ, đơn biến của dữ liệu, được gọi là "mô hình hồi quy tuyến tính đơn giản".

- - Video: [Mô hình hồi quy tuyến tính đơn giản](https://www.coursera.org/learn/ml-regression/lecture/N8p7w/the-simple-linear-regression-model)

Tuy nhiên do các tham số được khởi tạo một cách ngẫu nhiên hoặc là một giá trị liên tục nên cần một phương pháp đánh giá chất lượng giữa các tham số đó để đưa ra quyết định tham số nào đưa ra dự đoán tốt nhất. Video tiếp theo sẽ giải quyết vấn đề trên

- - Video: [Hàm chi phí](https://www.coursera.org/learn/ml-regression/lecture/WYPGc/the-cost-of-using-a-given-line)

Sau khi chọn được tham số w cho mô hình có chi phí thấp nhất, bạn sẽ học cách sử dụng tham số hoặc mô hình đó thông qua 2 video tiếp theo

- - Video: [Sử dụng đồ thị khớp](https://www.coursera.org/learn/ml-regression/lecture/RjYbf/using-the-fitted-line)
  - Video: [Diễn giải đồ thị khớp](https://www.coursera.org/learn/ml-regression/lecture/x8ohF/interpreting-the-fitted-line)



### 2. Tối ưu hàm chi phí

Trong Machine Learning nói riêng và Toán Tối Ưu nói chung, chúng ta thường xuyên phải tìm giá trị nhỏ nhất (hoặc đôi khi là lớn nhất) của một hàm số nào đó. Hướng tiếp cận phổ biến nhất là xuất phát từ một điểm mà chúng ta coi là gần với nghiệm của bài toán, sau đó dùng một phép toán lặp để tiến dần đến điểm cần tìm, tức đến khi đạo hàm gần với 0. Các điểm local minimum là nghiệm của phương trình đạo hàm bằng 0. Nếu bằng một cách nào đó có thể tìm được toàn bộ (hữu hạn) các điểm cực tiểu, ta chỉ cần thay từng điểm local minimum đó vào hàm số rồi tìm điểm làm cho hàm có giá trị nhỏ nhất (đoạn này nghe rất quen thuộc, đúng không?). Tuy nhiên, trong hầu hết các trường hợp, việc giải phương trình đạo hàm bằng 0 là bất khả thi. Hạ Gradient - Gradient Descent (viết gọn là GD) và các biến thể của nó là một trong những phương pháp được dùng nhiều nhất.

- Video: [Tìm cực đại/cực tiểu](https://www.coursera.org/learn/ml-regression/lecture/RUtxG/finding-maxima-or-minima-analytically)
- Video: [Ví dụ về tối đa hóa hàm 1D](https://www.coursera.org/learn/ml-regression/lecture/wN0CA/maximizing-a-1d-function-a-worked-example)
- Video: [Tìm cực đại bằng phương pháp leo đồi (hill climbing)](https://www.coursera.org/learn/ml-regression/lecture/O4j1e/finding-the-max-via-hill-climbing)
- Video: [Tìm cực tiểu bằng phương pháp xuống đồi (hill descent)](https://www.coursera.org/learn/ml-regression/lecture/zVcGn/finding-the-min-via-hill-descent)
- Video: [Chọn stepsize (kích thước bước nhảy) và tiêu chí hội tụ](https://www.coursera.org/learn/ml-regression/lecture/3UvFZ/choosing-stepsize-and-convergence-criteria)
- Video: [Gradients: Đạo hàm nhiều chiều](https://www.coursera.org/learn/ml-regression/lecture/ZwU5b/gradients-derivatives-in-multiple-dimensions)
- Video: [Hạ gradient: Xuống dốc đa chiều](https://www.coursera.org/learn/ml-regression/lecture/6PJ3h/gradient-descent-multidimensional-hill-descent)