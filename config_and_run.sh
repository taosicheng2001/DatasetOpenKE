!git clone https://github.com/thunlp/OpenKE.git
!cd OpenKE && cd openke && bash make.sh
!git clone https://github.com/tusenchun/DatasetOpenKE.git
!cp DatasetOpenKE -r OpenKE/
!cd OpenKE && cp DatasetOpenKE/transh_dataset.py ./ 
!cd OpenKE && mkdir checkopint && ls
!python3 transh_dataset