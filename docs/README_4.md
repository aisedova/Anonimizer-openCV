# Практика 4. Слежение за объектами через нахождение соответствий

## Цели

__Цель данной работы__ - реализовать трекинг объектов, который выполняется при помощи венгерского алгоритма, который используется для нахождения решения задачи о назначениях.

Алгоритм трекинга является достаточно сложным для того, чтобы его запрограммировать во время практики, поэтому в репозиторий `CV-SUMMER-CAMP` была включена готовая реализация от Леонида Бейненсона, которую он добавил в [`opencv_contrib`][opencv_extra_tracking]. Описание пошаговой разработки алгоритма тренига представлено в [ файле Google Docs][practice4_googledocs].

__Структура исходного кода__

   - include/tracking_by_matching.hpp - заголовочный файл алгоритма трекинга
   - src/tracking_by_matching.cpp - реализация алгоритма трекинга (использует задачу о назначениях)
   - src/kuhn_munkres.hpp - заголовочный файл алгоритма задачи  о назначениях 
   - src/kuhn_munkres.cpp - реализация алгоритма задачи  о назначениях 
   - samples/practice4.cpp - пример запуска трекинга

## Задачи

__Основные задачи:__

 1. Ознакомиться с исходным кодом трекинга.
 1. Скачать модель для детектирования объектов [mobilenet-ssd][mobilenetssd].
 1. Запустить пример `practice-4`, удостовериться в его работоспособности.
 1. Изменить код таким образом, чтобы можно было детектировать некоторое подмножество классов из всех доступных классов.

__Дополнительные задачи:__
 1. Реализовать матчинг объектов самостоятельно, при помощи алгоритма полного перебора, или написать свою версию венгерского алгоритма.
 1. Изменить код таким образом, чтобы в случае, когда нейросеть детектирует один объект на видео как несколько объектов разных классов, трекер считал этот объект одним классом.

## Общая последовательность действий

 1. в папке `<openvino_dir>`/deployment_tools/tools/model_downloader/  запустить скрипт downloader.py с параметрами --name mobilenet-ssd --output <destination_folder> 

        ```bash
        $ cd "C:\Intel\computer_vision_sdk\deployment_tools\tools\model_downloader"
        $ python downloader.py --name mobilenet-ssd --output <destination_folder>
        ```  
       В этой же папке расположен файл `list_topologies.yml`, в котором собраны параметры, перобразования входных изображений, они понаобятся для правильной конвертации вашей картинки.
 1. Ознакомиться с исходным кодом трекинга.
 1. Убедиться, что проект успешно собирается и создается новый исполняемый файл `<project_build>/bin/practice4.exe`.
 1. Запустить проект, указав все необходимые аргументы командной строки. В качестве тестового видеороклика использовать видео `catdog.mp4` из папки `data`.
 1. Повысить наглядность трекинга, изменить размер тип и размер шрифта, каждому классу назначить свой цвет, и т.д.
 1. В файле `practice4.cpp` в классе `DnnObjectDetector` изменить функцию `detect` таким образом, чтобы она могла детектировать только некоторое подмножество классов.
 1. В файле `practice4.cpp` изменить чтение параметра `desired_class_id` таким образом, чтобы можно было передавать массивы (например передавать строку, в которой имена классов разделены пробелом, и парсить эту строку). 
 1. Решить задачи из списка [Дополнительные задачи][addtasks]. 
 
 
<!-- LINKS -->
[mobilenetssd]: https://github.com/chuanqi305/MobileNet-SSD
[practice4_googledocs]: https://docs.google.com/document/d/1ebMY3juwGKqPhYSeU6drE68QVnjm7Ki3NvJcTYKOblg/edit
[opencv_extra_tracking]: https://github.com/opencv/opencv_contrib/tree/master/modules/tracking
[addtasks]: README_4.md#Задачи