import re
import os

def clean_text(input_file):
    # Проверяем существование входного файла
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден.")
        return
    
    # Открываем файл для чтения и записи
    with open(input_file, 'r+', encoding='utf-8') as infile:
        # Читаем весь файл в одну строку
        content = infile.read()
        
        # Удаляем слова, содержащие цифры
        content = re.sub(r'\b\w*\d\w*\b', '', content)  # Удаляем слова с цифрами
        
        # Заменяем все небуквенные символы на пробелы
        content = re.sub(r'[\W_]+', ' ', content)
        
        # Убираем лишние пробелы (несколько пробелов подряд)
        content = re.sub(r'\s+', ' ', content)
        
        # Убираем пробелы в начале и в конце строки
        content = content.strip()
        
        # Разбиваем текст на слова и фильтруем их по длине (от 4 до 13 символов)
        words = content.split()
        words = [word for word in words if 4 <= len(word) <= 13]
        
        # Собираем строку обратно из отфильтрованных слов
        content = ' '.join(words)
        
        # Перемещаем курсор в начало файла и перезаписываем его содержимое
        infile.seek(0)
        infile.write(content)
        infile.truncate()  # Убираем лишнее содержимое после текущей позиции курсора

# Пример использования
input_file = r'C:\code\SynthText-python3\data\newsgroup\newsgroup.txt'  # Имя файла, который будет изменен

clean_text(input_file)
