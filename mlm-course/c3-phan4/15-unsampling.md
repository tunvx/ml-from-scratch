# Undersampling

Trong bài học này, chúng ta sẽ tìm hiểu các phương pháp undersampling (Lấy mẫu dưới). Undersampling là quá trình giảm số lượng quan sát của lớp đa số. Vậy trong trường hợp undersampling điển hình, chúng ta sẽ thu được một tập dữ liệu có ít quan sát cửa lớp đa số hơn.

Chúng ta nên giảm tập dữ liệu đến mức nào hoặc khi nào thì dừng loại bỏ các mẫu khỏi tập dữ liệu? Có những phương pháp lấy mẫu dưới nào? Tại thời điểm nào thì nên sử dụng phương pháp nào? Liệu có một phương pháp mẫu tối ưu cho tất cả các tập dữ liệu? Hãy cùng nhau tìm câu trả lời trong bài học này.

- **Video:** [Giới thiệu về Undersampling](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859795#overview)

### 1. Fixed undersampling

Random undersampling trích xuất quan các sát ngẫu nhiên của lớp đa số hoặc các lớp đa số cho đến khi đạt được một tỷ lệ cân bằng nhất định. Đây là một kỹ thuật đơn giản, không giả định bất cứ điều gì về dữ liệu mà chỉ chọn các quan sát một cách ngẫu nhiên.

Khi quyết định áp dụng random undersampling, chúng ta cần cân nhắc một số điều. Mặt tốt là chúng ta thu được sự cân bằng tốt của mỗi một lớp. Mặt khác, bằng cách loại bỏ các quan sát khỏi lớp đa số hoặc các lớp đa số, chúng ta có thể đang loại các thông tin quan trọng khiến thuật toán khó tìm hiểu các mẫu để phân biệt các lớp hơn. Vì vậy, chúng ta cần kiểm tra xem random undersampling có cải thiện chất lượng mô hình hay không. Nếu không, chúng ta có thể áp dụng tiêu chí lấy mẫu khác.

- **Video:** [Random undersampling](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859799#overview)

Phương pháp NearMiss có 3 phiên bản khác nhau một chút nhưng về cơ bản, nó giữ lại những quan sát gần hơn với lớp thiểu số bằng cách này hay cách khác. NearMiss được thiết kế để làm việc với các tập dữ liệu văn bản, trong đó mỗi từ là một biểu diễn phức tạp của các từ và tag. Với các dữ liệu truyền thống hơn có biến dạng số và hạng mục, chúng ta cũng nên thử nghiệm.

- **Video:** [NearMiss](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859913#overview)

Instance Hardness là phép đo độ khó để phân loại trường hợp hoặc quan sát một cách chính xác. Instance hardness hay xác suất phân loại sai một quan sát thuộc vào hai điều. Đầu tiên là thuật toán mà chúng ta sử dụng để mô hình hóa nhiệm vụ và thứ hai là mối quan hệ quan sát với các quan sát khác trong tập dữ liệu, hay với chồng chéo lớp. 

Chúng ta có thể hiểu trực quan là một quan sát càng gần một quan sát của lớp đối diện thì thuật toán sẽ càng khó phân loại chúng một cách chính xác. Về cơ bản, các trường hợp hoặc quan sát khó phân loại chính xác là những trường hợp mà thuật toán học tập có xác suất dự đoán nhãn lớp đúng thấp. Vì vậy, với một quan sát của lớp thiểu số, xác suất thuật toán trả về càng thấp thì khả năng chúng bị phân loại sai càng cao, vậy đây là những trường hợp khó phân loại.

Phép đo instance hardness bằng một trừ đi xác suất và nó thể hiện xác suất của một quan sát bị phân loại sai. Vì vậy, khi xác suất lớp thấp thì instance hardness cao. Ngưỡng instance hardness là một ý tưởng rất đơn giản: loại bỏ các hard instance khỏi dữ liệu để giảm chồng chéo lớp và tăng khả năng phân tách lớp với các trường hợp có instance hardness lớn hơn một ngưỡng nhất định. 

Vậy vấn đề ở đây là làm thế nào để tìm ra ngưỡng này? Các tác giả của phương pháp này đã thử một số ngưỡng bất kỳ và họ kết luận rằng các ngưỡng khác nhau hoạt động tốt hơn với các tập dữ liệu khác nhau. Ưu điểm của kỹ thuật này là chúng ta có thể thay đổi ngưỡng để loại bỏ ít hoặc nhiều quan sát hơn.

- **Video:** [Ngưỡng Instance Hardness](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859927#overview)

### 2. Cleaning undersampling

Condensed Nearest Neighbours (CNN) trích xuất các quan sát của lớp đa số hoặc các lớp đa số với mục tiêu đa lớp nằm ở ranh giới giữa hai hoặc nhiều lớp trong tập dữ liệu, với giả thuyết rằng các thuật toán học máy nên tập trung vào các trường hợp khó để cải thiện chất lượng.

Những người đã sử dụng thành công mô hình này nói rằng các quan sát được chọn thường là những trường hợp khó nhất, chúng ta đang giúp thuật toán tập trung vào các trường hợp đó và tìm những mẫu đó giúp nó phân tách các lớp tốt hơn. Tuy nhiên, các trường hợp khó nhất đó thường rất khó phân loại. Vì vậy, có thể nói rằng kỹ thuật này tạo ra rất nhiều nhiễu. Cả hai khẳng định trên đều đúng. Vì vậy, hiệu quả của kỹ thuật này sẽ biến thiên theo tập dữ liệu mà chúng ta có.

- **Video:** [Condensed Nearest Neighbours](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859807#overview)

Nếu hai mẫu hoặc hai quan sát trong một tập dữ liệu là nearest neighbour của nhau và từ các lớp khác nhau thì chúng là các Tomek Link. Vậy nếu một quan sát của lớp đa số trông rất giống với một quan sát của lớp thiểu số thì đó là Tomek Link.

Trong quy trình loại bỏ, chúng ta có thể chỉ loại bỏ các mẫu đa số trong các cặp Tomek Link, hoặc loại cả cặp mẫu. 

Với Tomek Link, bằng cách loại bỏ nhiễu, chúng ta đang ngăn thuật toán máy học khỏi các trường hợp thực sự khó phân loại, và do đó bằng cách loại bỏ nhiễu, chúng ta có thể cải thiện chất lượng thuật toán. Nhưng nó cũng có thể khiến thuật toán phân loại sai các trường hợp khó hơn khi không có những trường hợp này ở ranh giới. Như thường lệ, chúng ta cần thử quy trình và xem nó hoạt động có hiệu quả hay không.

- **Video:** [Tomek Links](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859821#overview)

Ý tưởng về lựa chọn một bên là giữ lại các quan sát khó phân loại của lớp đa số và đồng thời loại bỏ nhiễu. Lựa chọn một bên có hai bước; ở bước đầu tiên, nó chọn các mẫu ở ranh giới của các lớp. Đây là những trường hợp khó nhất hoặc những quan sát khó phân loại đúng nhất. Sau đó nó tiếp tục và loại bỏ Tomek Link trong tập dữ liệu kết quả.

- **Video:** [Lựa chọn một bên](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859829#overview)

Ý tưởng của ENN là loại bỏ các mẫu gần ranh giới với các lớp khác nhất khỏi lớp đa số, do đó tăng cường phân tách của các lớp. ENN loại các quan sát có các neighbour không đồng ý với nó trong lớp, và thường có 3 neighbour được kiểm tra ở mỗi quan sát. Vì vậy, ENN loại bỏ các trường hợp khó phân loại nhất vì chúng gần ranh giới với các lớp khác nhất.

- **Video:** [Edited Nearest Neighbours](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859841#overview)

RENN là phần mở rộng của Edited Nearest Neighbour mà chúng ta đã thảo luận trong video trước. RENN sẽ thực hiện lặp đi lặp lại ENN cho đến khi không còn quan sát nào bị loại nữa hoặc cho tới khi đạt tới số chu kỳ tối đa.

- **Video:** [Repeated Edited Nearest Neighbours](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859861#overview)

Ý tưởng của ENN là loại bỏ các mẫu gần ranh giới với các lớp khác nhất khỏi lớp đa số. All KNN là phần mở rộng của thuật toán ENN mà chúng ta đã thảo luận trong video trước. Nó lặp lại ENN một số lần nhất định. Nó bắt đầu bằng cách khám phá một neighbour gần nhất ở lần lặp đầu tiên, sau đó nó thêm một neighbour vào KNN ở mỗi lần lặp. 

Thuật toán sẽ quyết định giữ hay loại quan sát khỏi lớp đa số dựa trên sự thống nhất của neighbour với lớp. Nó sẽ dừng lại khi đạt được số lượng neighbour tối đa do người dùng quyết định hoặc khi lớp đa số trở thành lớp thiểu số.

Nhìn chung, All KNN - cũng như RENN sẽ tự loại bỏ nhiều mẫu hơn so với ENN vì nó thực hiện nhiều lần truyền qua dữ liệu và các KNN kế tiếp có nhiều neighbour hơn so với KNN ở vòng trước.

- **Video:** [All KNN](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859875#overview)

Ý tưởng của quy tắc Neighbourhood Cleaning là loại bỏ các mẫu gần với ranh giới các lớp khác nhất khỏi (các) lớp đa số, tăng cường sự phân tách của các lớp và loại bỏ nhiễu. 

Nó mở rộng dựa trên thuật toán ENN. Đầu tiên, chúng ta thực thi ENN như thông thường. Bước tiếp theo trong NCL sau ENN là bước làm sạch - tập trung vào các neighbour của những quan sát từ lớp thiểu số. Với mỗi quan sát của thiểu số, nó sẽ tìm 3 neighbour gần nhất. Và nếu tất cả hoặc hầu hết các neighbour không đồng ý với lớp thiểu số thì các neighbour này sẽ bị loại bỏ. Có một ngoại lệ với quy tắc này là nếu các neighbour không đồng ý với lớp, nhưng chúng cũng thuộc lớp thiểu số khác thì chúng sẽ không bị loại bỏ.

- **Video:** [Quy tắc Neighbourhood Cleaning](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859897#overview)

Chúng ta cần nhớ rằng các kỹ thuật khác nhau sẽ hoạt động tốt hơn trong các tập dữ liệu khác nhau. Vì vậy, chúng ta sẽ cần thử nghiệm rất nhiều để biết kỹ thuật nào sẽ cải thiện chất lượng mô hình. Chúng ta có thể sử dụng rất nhiều công cụ để rút ra chất lượng từ các mô hình học máy được xây dựng trên tập dữ liệu mất cân bằng.

- **Video:** [So sánh các phương pháp undersampling](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/22859941#overview)

### Lab 13: Undersampling với pandas, sklearn và imblearn

- [Notebooks](https://drive.google.com/drive/folders/1rZRqJlKrxouX4zWLCKnIpIW3PgogPvaz?usp=share_link)