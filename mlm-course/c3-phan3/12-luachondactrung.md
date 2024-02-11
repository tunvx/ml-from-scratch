# Lựa chọn đặc trưng dựa theo cây quyết định

Làm thế nào để lựa chọn đặc trưng bằng độ quan trọng của cây? Khi huấn luyện một cây, có thể tính toán xem mỗi đặc trưng làm giảm bao nhiêu nhiễu hay nói cách khác là đặc trưng tách các lớp tốt như thế nào. Đặc trưng càng giảm nhiều nhiễu thì càng quan trọng. 

Random forest gồm 400 - 1200 DT. Mỗi cây trong số đó được xây dựng dựa trên việc trích xuất ngẫu nhiên các quan sát từ tập dữ liệu và trích xuất ngẫu nhiên các đặc trưng. Không phải cây nào cũng thấy tất cả các đặc trưng hay tất cả các quan sát và điều này đảm bảo rằng các cây giảm tương quan, do đó ít bị overfitting hơn. 

Trong random forest, nhiễu giảm từ mỗi đặc trưng có thể là trung bình trên tất cả các cây để xác định độ quan trọng cuối cùng của biến. Điều này chỉ đúng với random forest và không chính xác với các thuật toán boosing như gradient boosing (bạn sẽ được học kỹ hơn về boosting ở bài 17).

- ***\*Video: [Tree Importance | Giới thiệu](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341792#overview)\****

Video ngắn này sẽ hướng dẫn cách lựa chọn đặc trưng bằng cách sử dụng độ quan trọng của Random Forest. Chúng ta sẽ được làm quen với một chức năng rất quan trọng trong sklearn - SelectFromModel.

- ***\*Video: [Tree Importance | Demo](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22609692#overview)\****

Chúng ta có thể chọn đặc trưng một cách đệ quy. Để lựa chọn đặc trưng một cách đệ quy, trước tiên chúng ta sử dụng tất cả các đặc trưng để xây dựng một random forest, lấy độ quan trọng của các đặc trưng này. Sau đó, chúng ta loại bỏ đặc trưng ít quan trọng nhất và giờ chúng ta xây dựng một random forest mới với các đặc trưng còn lại và tính toán lại độ quan trọng. Chúng ta lặp lại quá trình này cho đến khi đáp ứng một số tiêu chí nhất định.

- ***\*Video:\** [\**Tree Importance | Đệ quy\**](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22609702#overview)**

### Labs

- [Lab 11.1: Lựa chọn đặc trưng cho bài toán phân loại](https://drive.google.com/drive/folders/18N9ssq-pAozEZN8Z_3VBKMADEVKwShOK?usp=share_link)
- [Lab 11.2: Lựa chọn đặc trưng dựa theo cây quyết định](https://drive.google.com/drive/folders/12sHlGJ8ABiPGuDrFo9OqROF3BMr2z6Kl?usp=share_link)
- [Dataset](https://drive.google.com/file/d/1R7gTeQH59cKXi6vw4dOiOj6cv0-mRqb3/view?usp=share_link)