# Ensemble with imbalenca data

Trong bài học trước, Bagging đề cập đến việc tạo các mẫu khác với tập dữ liệu gốc bằng Bootstrapping có hoàn lại. Sau đó ở mỗi tập dữ liệu này, chúng ta huấn luyện một bộ phân loại khác rồi kết hợp các bộ phân loại để đưa ra dự đoán cuối cùng. 

Phương pháp UnderBagging kết hợp sử dụng Random Under-sampling với Bagging. Trong trường hợp này, chúng ta cần tạo một bản sao cho mỗi tập dữ liệu của lớp thiểu số, rồi boostrap có hoàn trả số lượng quan sát của lớp đa số bằng số lượng quan sát của lớp thiểu số. Sau đó, trong các tập dữ liệu cân bằng này, chúng ta huấn luyện thuật toán rồi kết hợp các dự đoán về thuật toán để đưa ra dự đoán cuối cùng. Nếu thuật toán chính được sử dụng là Decision Tree, thì thuật toán tổng thể được gọi là balanced random forest.

Chúng ta cũng có thể kết hợp Bagging với Random Over-sampling, gọi là OverBagging. Trong trường hợp này, từ tập dữ liệu ban đầu, chúng ta boostrap có hoàn trả cả quan sát từ lớp đa số và lớp thiểu số. Thay vì oversampling, chúng ta cũng có thể sử dụng SMOTE để tạo tập dữ liệu huấn luyện bộ phân loại, gọi là SMOTEBagging. 

- ***\*Video: [Bagging với Over hay Under-Sampling](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/24339952#overview)\****

Trong RUSBoost hay Random Undersampling + Boosting, chúng ta kết hợp Random Undersampling với Boosting. Ở lần lặp đầu tiên, chúng ta chỉ định các trọng số bằng nhau cho mỗi quan sát, các trọng số này được tính bằng 1 chia cho số lượng quan sát. Khi mỗi mẫu có trọng số được chỉ định, chúng ta undersample ngẫu nhiên tập dữ liệu ban đầu. Sau đó, chúng ta huấn luyện một bộ phân loại ở dạng dữ liệu undersampled này, xem xét trọng số của từng quan sát. Sau khi đã huấn luyện bộ phân loại, chúng ta sẽ tính toán lỗi trên toàn bộ tập dữ liệu, không chỉ trên các mẫu mẫu undersampled. Với lỗi, chúng ta điều chỉnh trọng số cho tất cả các quan sát. Sau đó, chúng ta lại lấy một undersample ngẫu nhiên khác, huấn luyện bộ phân loại thứ hai, tính toán lỗi, điều chỉnh trọng số,...

SMOTEBoost rất giống với RUSBoost, nó kết hợp SMOTE với Boosting. Với SMOTEBoost, chúng ta sẽ tạo ra nhiều thực thể hơn của lớp thiểu số. Như vậy, chúng ta đang cải thiện độ chính xác của bộ phân loại bằng cách cung cấp nhiều mẫu của lớp thiểu số hơn cho bộ phân loại, nhưng khi tạo mẫu mới, chúng ta cũng đang tạo ra sự đa dạng để các bộ phân loại tiếp theo đa dạng hơn.

RAMOBoost rất giống với SMOTEBoost, nhưng thay vì sử dụng SMOTE để tạo mẫu tổng hợp, RAMOBoost lại sử dụng một kỹ thuật rất giống với ADASYN.

- ***\*Video: [Boosting với Re-Sampling](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/24396064#overview)\****

Đặc điểm của Bagging + Boosting + Re-sampling là chúng ta huấn luyện thuật toán Boosting ở mỗi bag, và các bag được tạo ra nhờ random undersampling hoặc oversampling.

Cụ thể hơn, khi xem xét các tập dữ liệu mất cân bằng, chúng ta thường có một lớp undersampling khi tạo bag từ dữ liệu gốc. Sau đó, chúng ta sẽ huấn luyện một thuật toán Adaboost từ các Bag này. Phương pháp này còn được gọi là thuật toán EasyEnsemble.

BalanceCascade khá giống với EasyEnsemble, chỉ có 1 điều chỉnh nhỏ là các Adaboost sẽ được huấn luyện tuần tự, thay vì song song. Chúng ta tạo bag đầu tiên bằng cách sao chép tất cả các quan sát của lớp thiểu số và bootstrap có hoàn trả một số quan sát của lớp đa số, sau đó chúng ta huấn luyện AdaBoost trong bag đầu tiên. AdaBoost phân loại chính xác nhiều quan sát của lớp đa số. Sau đó, chúng ta loại những quan sát được AdaBoost phân loại đúng của lớp đa số ra khỏi tập dữ liệu ban đầu. Chúng ta trích xuất một bag khác, xây dựng một AdaBoost khác, tiến hành lặp đi lặp lại số lượng estimator mà chúng ta muốn cho mẫu cuối cùng.

- ***\*Video: [Các phương pháp lai hóa](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/24396066#overview)\****