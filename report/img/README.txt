Сюда кладутся все картинки для отчёта.

Используй в .tex файлах:
  \includegraphics[width=0.8\linewidth]{имя_файла.png}

Путь прописан в preamble.tex через \graphicspath{{img/}},
поэтому писать img/ в \includegraphics не нужно.

Примеры ожидаемых файлов:
  pipeline.png      -- схема цепочки симуляции (макропроект)
  macro_result.png  -- результат из ParaView
  micro_result.png  -- результат микропроекта
