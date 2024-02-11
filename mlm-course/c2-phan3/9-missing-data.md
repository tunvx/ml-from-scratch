# Xử lý dữ liệu bị khuyết



**Video: [Xử lý dữ liệu bị khuyết (missing data)](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/15675348#overview)**



## 1. CCA

- **Video: [Phân tích trường hợp toàn vẹn](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/7836454#overview)**



## 2. Gán giá trị cho dữ liệu khuyết

Gán giá trị trung bình/giá trị trung vị bao gồm việc thay thế tất cả các lần xuất hiện giá trị bị thiếu (NA) trong một biến bằng giá trị trung bình (nếu biến có phân phối Gauss) hoặc trung vị (nếu biến có phân phối lệch).

- **Video: [Gán giá trị trung bình hay trung vị](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/7690440#overview)**



Gán giá trị bất kỳ bao gồm việc thay thế tất cả các lần xuất hiện giá trị bị thiếu (NA) trong một biến bằng một giá trị tùy ý. Các giá trị tùy ý thường được sử dụng là 0,999, -999 (hoặc các kết hợp khác của 9) hoặc -1 (nếu phân phối là dương).

- **Video: [Gán giá trị bất kỳ](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8669706#overview)**



Việc xác định giá trị của giá trị tùy ý có thể tốn nhiều công sức và nó thường là một công việc thủ công. Chúng ta có thể tự động hóa quá trình này bằng cách tự động chọn các giá trị tùy ý ở cuối các bản phân phối biến.

- **Video: [Gán giá trị ở cuối phân phối](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8612792#overview)**



Gán hạng mục thường xuất hiện bao gồm việc thay thế tất cả các lần xuất hiện của các giá trị bị thiếu (NA) trong một biến bằng giá trị mode. Nói cách khác, nó đề cập đến giá trị thường xuyên nhất hoặc danh mục thường xuyên nhất.

- **Video: [Gán hạng mục thường xuất hiện](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8670358#overview)**



Coi dữ liệu bị thiếu như một hạng mục mới là phương pháp được sử dụng rộng rãi nhất để xử lý dữ liệu bị thiếu cho các biến phân loại. Phương pháp này bao gồm việc xử lý dữ liệu bị thiếu như một nhãn hoặc danh mục bổ sung của biến. Tất cả các quan sát bị thiếu được nhóm lại trong nhãn 'Thiếu' mới được tạo.

Về bản chất, điều này tương đương với việc thay thế bằng một giá trị tùy ý cho các biến số. Ưu điểm của kỹ thuật này nằm ở thực tế là nó không giả định bất cứ điều gì về thực tế là dữ liệu bị thiếu. Nó rất phù hợp khi số lượng dữ liệu bị thiếu nhiều.

- **Video: [Dữ liệu bị khuyết là một hạng mục](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8748548#overview)**



Việc áp dụng gán mẫu ngẫu nhiên về nguyên tắc tương tự như việc áp dụng giá trị trung bình/trung vị/hạng mục thường xuất hiện, theo nghĩa là nó nhằm mục đích bảo toàn các tham số thống kê của biến ban đầu, mà dữ liệu bị thiếu. Gán mẫu ngẫu nhiên bao gồm việc lấy một quan sát ngẫu nhiên từ nhóm các quan sát có sẵn về biến và sử dụng giá trị được trích xuất ngẫu nhiên đó để thay thế NA.

- **Video: [Gán mẫu ngẫu nhiên](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8659914#overview)**



Có những phương pháp khác có thể được sử dụng để gán khi các giá trị không bị thiếu một cách ngẫu nhiên, ví dụ: gán giá trị bất kỳ hoặc lấy ở cuối phân phối. Tuy nhiên, các kỹ thuật áp đặt này sẽ ảnh hưởng đáng kể đến phân phối biến số, do đó không phù hợp với các mô hình tuyến tính. 

Nếu dữ liệu không bị thiếu một cách ngẫu nhiên, bạn nên thay thế các quan sát bị thiếu bằng giá trị trung bình/ trung vị/hạng mục thường xuất hiện và gắn cờ các quan sát bị thiếu đó cũng như bằng một Chỉ báo bị thiếu. Chỉ báo Thiếu là một biến nhị phân bổ sung, cho biết liệu dữ liệu có bị thiếu cho một quan sát (1) hay không (0).

- **Video: [Chỉ số khuyết dữ liệu](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8612158#overview)**



## 3. Gán đa biến

- Video: [Gán đa biến](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/24165968#overview)

- Video: [Sử dụng KNN](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/24165978#overview)
- Video: [MICE](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/24165986#overview)

- Video: [missForest](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/24165988#overview)



## 4. Lab 8: Gán dữ liệu bị khuyết với Pandas

- [Dataset](https://drive.google.com/file/d/17QvEIOCNPUz3WHdvf3vwbHoDj_O5w8Hd/view)
- [Notebooks](labs/lab8.zip)

