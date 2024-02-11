# Xử lý ngôn ngữ tự nhiên với Python

Chào mừng các bạn đến với bài học về xử lý ngôn ngữ tự nhiên với Python. Mục tiêu của bài học này là thiết lập Spacy và thư viện ngôn ngữ mà chúng ta cần download cho Spacy. Chúng ta sẽ tìm hiểu một số chủ đề NLP cơ bản, gồm Tokenization, Stemming, Lemmatization, Stop Words và sử dụng Spacy cho Vocabulary Matching. Chúng ta sẽ có một vài bài giảng giới thiệu để thảo luận về các thư viện phổ biến như NLTK với Spacy và lý do chúng ta sử dụng cả hai trong khóa học này cũng như thảo luận chung về cách xử lý ngôn ngữ tự nhiên.

- ***\*Video: [Giới thiệu về xử lý ngôn ngữ tự nhiên](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827885#reviews)\****

### 1. Tổng quan về NLP

Thường khi thực hiện một số phân tích, có rất nhiều dữ liệu ở dạng số và điều đó thực sự thuận tiện như doanh số bán hàng, phép đo vật lý, hạng mục có thể định lượng. Máy tính rất giỏi trong việc xử lý thông tin số trực tiếp. Tuy nhiên, chúng ta có thể làm gì với máy tính về dữ liệu ngôn ngữ tự nhiên/thông tin văn bản? NLP cố gắng sử dụng nhiều kỹ thuật khác nhau để tạo ra một số loại cấu trúc từ dữ liệu văn bản thô, cụ thể là cách lập trình máy tính để xử lý và phân tích lượng lớn dữ liệu ngôn ngữ tự nhiên.

- **Video: [Xử lý ngôn ngữ tự nhiên là gì?](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827889#reviews)**

Trước hết, chúng ta cùng tìm hiểu qua về thư viện Spacy.

- **Video: [Cài đặt Spacy](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827887#reviews)**

Cùng nhau khám phá pipeline object cho strings theo các hoạt động khác nhau.

- **Video: [Spacy cơ bản](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827891#reviews)**



### 2. Các thao tác cơ bản với NLP

Trong Xử lý ngôn ngữ tự nhiên, Tokenization là quá trình chuyển một dãy các ký tự thành một dãy các token (token là một dãy các ký tự mang ý nghĩa cụ thể, biểu thị cho một đơn vị ngữ nghĩa trong xử lý ngôn ngữ). Nhiều khi token được hiểu là một từ mặc dù cách hiểu này không hoàn toàn chính xác.

- **Video: [Tokenization - Phần 1](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827893#reviews)**

- **Video: [Tokenization - Phần 2](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827899#reviews)**

Trong quá trình xử lý ngôn ngữ tự nhiên, chúng ta sẽ có nhu cầu so sánh các từ (token) với nhau. Việc so sánh này tưởng chừng như đơn giản là lấy 2 chuỗi ký tự và dùng phép “==” để kiểm tra, nhưng thực tế thì không phải là như vậy. Đối với một số ngôn ngữ, tiêu biểu là tiếng Anh, mỗi từ có thể có nhiều biến thể khác nhau. Điều này làm cho việc so sánh giữa các từ là không thể mặc dù về mặc ý nghĩa cơ bản là như nhau. Ví dụ các từ “walks“, “walking“, “walked” đều là các biến thể của từ “walk” và đều mang ý nghĩa là “đi bộ”. Vậy làm sao để so sánh các từ như thế với nhau? Lemmatization và Stemming chính là 2 kỹ thuật thường được dùng cho việc này.

Stemming là kỹ thuật dùng để biến đổi 1 từ về dạng gốc (được gọi là stem hoặc root form) bằng cách cực kỳ đơn giản là loại bỏ 1 số ký tự nằm ở cuối từ mà nó nghĩ rằng là biến thể của từ. Ví dụ như chúng ta thấy các từ như **walked**, **walking**, **walks** chỉ khác nhau là ở những ký tự cuối cùng, bằng cách bỏ đi các hậu tố –**ed**, –**ing** hoặc –**s**, chúng ta sẽ được từ nguyên gốc là **walk**. Người ta gọi các bộ xử lý

- **Video: [Stemming](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827901#reviews)**

Khác với Stemming là xử lý bằng cách loại bỏ các ký tự cuối từ một cách rất heuristic, Lemmatization sẽ xử lý thông minh hơn bằng một bộ từ điển hoặc một bộ ontology nào đó. Điều này sẽ đảm bảo rằng các từ như “**goes**“, “**went**” và “**go**” sẽ chắc chắn có kết quả trả về là như nhau. Kể các từ danh từ như **mouse**, **mice** cũng đều được đưa về cùng một dạng như nhau.

- **Video: [Lemmatization](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827905#reviews)**

### 3. Các thao tác nâng cao với NLP

Stopword à những từ trong bất kỳ ngôn ngữ nào không bổ sung nhiều ý nghĩa cho một câu. Chúng có thể được bỏ qua một cách an toàn mà không làm mất đi ý nghĩa của câu. Đối với một số công cụ tìm kiếm, đây là một số từ chức năng ngắn, phổ biến nhất, chẳng hạn như, **is, at, which,** và **on**. Trong trường hợp này, các từ dừng có thể gây ra vấn đề khi tìm kiếm các cụm từ bao gồm chúng, đặc biệt là trong các tên như “**The Who”** hoặc **“Take That”.**

- **Video: [Stop Words](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827907#reviews)**

Cùng tìm hiểu thêm về các kỹ thuật nâng cao khác trong NLP:

- **Video tham khảo: [Phrase Matching and Vocabulary - Phần 1](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827911#reviews)**
- **Video tham khảo: [Phrase Matching and Vocabulary - Phần 2](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827913#reviews)**



### Lab 4 - Xử lý text với Python

Chạy, đọc, làm theo các hướng dẫn trong notebook và điền vào tất cả các khối code trống cần thiết. Đảm bảo hoàn thành mọi thứ và trả lời tất cả các câu hỏi thiết yếu cho các lab sau:

- Lab 4.1: **Xử lý text cơ bản**
- Lab 4.2: **NLP cơ bản**

Các bạn download dữ liệu và notebook về tại [đây](https://drive.google.com/drive/folders/1qYFZe-H5yVHh9vFAtHUfcy7aiX61oHbP?usp=sharing).

Nếu các bạn không hiểu được các đoạn code trong lab hay không biết giải quyết các vấn đề trong bài lab ra sao, có thể tham khảo phần lý giải chi tiết trong các video dưới đây:

- ***\*[Lab 4.1](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827881#reviews)\****
- ***\*[Lab 4.2](https://funix.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/12827917#reviews)\****