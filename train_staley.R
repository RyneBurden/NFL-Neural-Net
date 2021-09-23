reticulate::use_virtualenv("C:/Users/Ryne/Documents/R Projects and Files/Staley_2.0/virtualenvs/staley")
reticulate::py_run_file("staley_train-v2.py")

num_epochs = readline("How many training epochs?")

reticulate::py$net_main(reticulate::r_to_py(training_data %>% filter(!is.na(OFF_RUSH_EPA))), .000015, as.integer(num_epochs))

remove(num_epochs)