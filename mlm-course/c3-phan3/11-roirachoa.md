# Rời rạc hóa

## 1. Rời rạc hóa không giám sát

Rời rạc hóa sử dụng khoảng cách bằng nhau chia phạm vi giá trị có thể của biến thành N bin hoặc các khoảng (interval). 

Các khoảng/bin này có khoảng cách (width) như nhau; khoảng cách được xác định bởi giá trị lớn nhất của biến, giá trị nhỏ nhất của biến và số khoảng mà chúng ta muốn tạo.

- **Video: [Rời rạc hóa sử dụng khoảng cách bằng nhau](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16165849#overview)**

Rời rạc hóa sử dụng tần số bằng nhau chia phạm vi các giá trị có thể của biến thành một số khoảng mà mỗi khoảng lại chứa số lượng quan sát xấp xỉ nhau. Thông thường, để tính ranh giới cho từng khoảng này, chúng ta sẽ tính các quantile của biến.

Kỹ thuật này giúp cải thiện chênh lệch giá trị và nó cũng xử lý outlier khi chúng được phân bổ vào vùng đầu hoặc cuối. Việc cải thiện chênh lệch giá trị sẽ giúp các mô hình tuyến tính giả định chênh lệch đều hơn hoặc phân phối các giá trị của biến chuẩn hơn.

- **Video: [Rời rạc hóa sử dụng tần số bằng nhau](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16165877#overview)**

Rời rạc hóa sử dụng K-means gồm việc áp dụng phân cụm K-means vào biến liên tục để thu được các cụm khác nhau, mỗi cụm tương ứng với một bin mà chúng ta sẽ sắp xếp các giá trị của biến. 

Rời rạc hóa sử dụng K-means không cải thiện chênh lệch giá trị. Nó giúp xử lý outlier, mặc dù outlier có thể có ảnh hưởng đến vị trí tâm của các cụm. 

- **Video: [Rời rạc hóa sử dụng K-means](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16165893#overview)**

Trong video này, chúng ta sẽ thảo luận cách sử dụng biến sau khi đã rời rạc hóa. Chúng ta đã đề cập rằng rời rạc hóa là quá trình chia một khoảng liên tục thành một tập hợp các khoảng kéo dài phạm vi giá trị. Nếu sử dụng nó ở dạng số, chúng ta có thể sử dụng trực tiếp giá trị của khoảng, 

Nhưng nếu sử dụng nó ở dạng hạng mục, chúng ta cần áp dụng bất kỳ mã hóa nào mà chúng ta đã thấy trong phần mã hóa hạng mục vào biến dạng số ban đầu này (đã được đề cập ở môn MLP302x, bài học số 10). Và trên thực tế, cách hữu ích để mã hóa các bin này là sử dụng bộ mã hóa tạo ra mối quan hệ đơn điệu giữa bin với mục tiêu.

- ***\*Video: [Rời rạc hóa kết hợp mã hóa hạng mục](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16165901#overview)\****

## 1. Rời rạc hóa không giám sát

 Bookmark this page

Rời rạc hóa sử dụng khoảng cách bằng nhau chia phạm vi giá trị có thể của biến thành N bin hoặc các khoảng (interval). 

Các khoảng/bin này có khoảng cách (width) như nhau; khoảng cách được xác định bởi giá trị lớn nhất của biến, giá trị nhỏ nhất của biến và số khoảng mà chúng ta muốn tạo.

- **Video: [Rời rạc hóa sử dụng khoảng cách bằng nhau](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16165849#overview)**

Rời rạc hóa sử dụng tần số bằng nhau chia phạm vi các giá trị có thể của biến thành một số khoảng mà mỗi khoảng lại chứa số lượng quan sát xấp xỉ nhau. Thông thường, để tính ranh giới cho từng khoảng này, chúng ta sẽ tính các quantile của biến.

Kỹ thuật này giúp cải thiện chênh lệch giá trị và nó cũng xử lý outlier khi chúng được phân bổ vào vùng đầu hoặc cuối. Việc cải thiện chênh lệch giá trị sẽ giúp các mô hình tuyến tính giả định chênh lệch đều hơn hoặc phân phối các giá trị của biến chuẩn hơn.

- **Video: [Rời rạc hóa sử dụng tần số bằng nhau](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16165877#overview)**

Rời rạc hóa sử dụng K-means gồm việc áp dụng phân cụm K-means vào biến liên tục để thu được các cụm khác nhau, mỗi cụm tương ứng với một bin mà chúng ta sẽ sắp xếp các giá trị của biến. 

Rời rạc hóa sử dụng K-means không cải thiện chênh lệch giá trị. Nó giúp xử lý outlier, mặc dù outlier có thể có ảnh hưởng đến vị trí tâm của các cụm. 

- **Video: [Rời rạc hóa sử dụng K-means](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16165893#overview)**

Trong video này, chúng ta sẽ thảo luận cách sử dụng biến sau khi đã rời rạc hóa. Chúng ta đã đề cập rằng rời rạc hóa là quá trình chia một khoảng liên tục thành một tập hợp các khoảng kéo dài phạm vi giá trị. Nếu sử dụng nó ở dạng số, chúng ta có thể sử dụng trực tiếp giá trị của khoảng, 

Nhưng nếu sử dụng nó ở dạng hạng mục, chúng ta cần áp dụng bất kỳ mã hóa nào mà chúng ta đã thấy trong phần mã hóa hạng mục vào biến dạng số ban đầu này (đã được đề cập ở môn MLP302x, bài học số 10). Và trên thực tế, cách hữu ích để mã hóa các bin này là sử dụng bộ mã hóa tạo ra mối quan hệ đơn điệu giữa bin với mục tiêu.

- ***\*Video: [Rời rạc hóa kết hợp mã hóa hạng mục](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16165901#overview)\****



## Lab 10: Rời rạc hóa

- [Notebooks](https://drive.google.com/drive/folders/1H1UdDXZbr0V0H15Hp7gXEw8GknLlbCTQ?usp=share_link)
- [Dataset](https://drive.google.com/file/d/12hYXYE2IrxIOlE1w74BuFVNmqk7gB57W/view?usp=share_link)