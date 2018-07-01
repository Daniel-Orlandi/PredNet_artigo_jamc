echo "Começando treinamento"
source activate tensorflow
cd /home/daniel/AnacondaProjects/prednet-master_2/py_scripts
while true
do
  # (1) prompt user, and read command line argument
  read -p "Rodar o kitti_train.py agora? " answer

  # (2) handle the input we were given
  case $answer in
   [yY]* ) echo "Ok, rodando o script."
           python kitti_train.py
           
           break;;

   [nN]* ) break;;

   * )     echo "Digite apenas Y ou N.";;
  esac
done

while true
do
  # (1) prompt user, and read command line argument
  read -p "Rodar o kitti_extrap_finetune.py agora? " answer

  # (2) handle the input we were given
  case $answer in
   [yY]* ) echo "Ok, rodando o script."
           python kitti_extrap_finetune.py
           
           break;;

   [nN]* ) break;;

   * )     echo "Digite apenas Y ou N.";;
   esac
done


echo 'Treinamento completado =)'