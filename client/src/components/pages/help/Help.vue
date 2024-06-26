<template>
  <div class="container m-4">
    <div class="card">
      <div class="card-body">
        <h4 class="text-center">Визуализатор обучения нейронных сетей</h4>
        <div id="accordion" class="accordion mt-2">
          <div class="accordion-item">
            <h2 class="accordion-header">
              <button
                class="accordion-button collapsed"
                type="button"
                data-bs-toggle="collapse"
                data-bs-target="#help0"
              >
                Базовая информация о графовом представлении
              </button>
            </h2>
            <div id="help0" class="accordion-collapse collapse">
              <div class="accordion-body">
                <p>
                  Выбранная нейронная сеть представлена в прямоугольнике, сверху от которого
                  написано имя нейронной сети и текущее значение функции потерь, которое будет
                  обновляться в конце каждого батча.
                </p>
                <p>
                  Батч - фрагмент от исходного датасета данных, на которых обучается нейронная сеть.
                  Батч используется для одного продвижения весов нейронной сети с помощью
                  градиентного спуска к значениям, которые минимизируют функцию потерь.
                </p>
                <p>Эпоха - это полное прохождение тренировочного датасета через нейронную сеть</p>
                <p>
                  Функция потерь - функция, которая характеризует потери при неправильном принятии
                  решений на основе наблюдаемых данных. Т.е. грубо говоря, это численная мера оценки
                  того насколько неверный результат был получен на выходе нейронной сети.
                </p>
                <p>
                  Входные данные представлены первым слоем графового представления нейронной сети,
                  выходные данные (результат работы всей нейронной сети) - самый последний слой.
                </p>
                <p>
                  В качестве упрощения прохождение через нейронную сеть каждого 1 набор данных из
                  датасета показывается отдельно. Хотя в реальности весь батч наборов данных
                  обрабатывается одновременно за счет векторных операций.
                </p>
                <p>
                  Обучение нейронной сети разделяется на "прямой" и "обратный" проход.
                  <br />"Прямой" проход - это обычное изменение входных данных при прохождении по
                  нейронной сети с вычислением функции потерь (насколько нейронная сеть ошибается),
                  когда все данные в батче уже прошли через нейронную сеть и функция потерь
                  посчитана наступает второй этап - "обратный" проход (он же называется обратным
                  распространением ошибки). <br />Во втором этапе происходит обновление весов
                  нейронной сети таким образом, чтобы приблизиться к минимизации функции потерь для
                  текущего батча на заданный шаг обучения (learning rate).
                </p>
              </div>
            </div>

            <h2 class="accordion-header">
              <button
                class="accordion-button collapsed"
                type="button"
                data-bs-toggle="collapse"
                data-bs-target="#help1"
              >
                Элементы графового представления нейронных сетей
              </button>
            </h2>
            <div id="help1" class="accordion-collapse collapse">
              <div class="accordion-body">
                <p>
                  Квадратные ячейки - это просто данные, которые видоизменяются проходя через
                  нейронную сеть.
                </p>
                <p>
                  Linear - это "нейрон", можно нажать на данный элемент в графовом представлении
                  нейронной сети, чтобы посмотреть веса, которые имеют данные, проходящие через
                  данный нейрон.
                </p>
                <p>
                  Activation - это некоторая функция, которая вносит нелинейность в преобразование
                  данных "нейронами" нейронной сети.
                </p>
                <p>
                  Conv2d - это сверточный фильтр, который используется для получения информации о
                  связи близко расположенных данных в нейронной сети.
                </p>
                <p>
                  MaxPool2d - это фильтр, который используется для уменьшения размера двумерных
                  данных путем сворачивания некоторой области (например области 2 на 2) в 1 значение
                  соответствующее максимальному значению в данной области.
                </p>
                <p>
                  MergeFlatten - это операция склеивания различных наборов двумерных данных друг с
                  другом с последующим превращением в одномерный вектор.
                </p>
                <p>Reshape - это операция превращения одномерных данных в двумерные.</p>
              </div>
            </div>
          </div>

          <div class="accordion-item">
            <h2 class="accordion-header">
              <button
                class="accordion-button collapsed"
                type="button"
                data-bs-toggle="collapse"
                data-bs-target="#help2"
              >
                Small ANN (Маленькая версия полносвязной нейронной сети)
              </button>
            </h2>
            <div id="help2" class="accordion-collapse collapse">
              <div class="accordion-body">
                <p>
                  Описание: маленький упрощенный вариант стандартной полносвязной нейронной сети для
                  знакомства с процессом обучения на очень тренировочной задаче - попытка обучить
                  нейронную сеть для работы в качестве логической операции XOR. На вход подаются 2
                  значения, каждое из которых либо 0, либо 1. Ожидаемый результат - вероятность
                  того, что результат XOR будет 0 или 1.
                </p>
                <p>
                  Датасет: таблица истинности для XOR. Состоит из всего 4 пар (0, 0), (0, 1), (1,
                  0), (1, 1). Ожидаемый результат - 0, 1, 1, 0 для входных пар соответственно (т.е.
                  нейронная сеть должна выдать бОльшую вероятность, чем для другого значения).
                </p>
              </div>
            </div>
          </div>

          <div class="accordion-item">
            <h2 class="accordion-header">
              <button
                class="accordion-button collapsed"
                type="button"
                data-bs-toggle="collapse"
                data-bs-target="#help3"
              >
                ANN (Полносвязная нейронная сеть)
              </button>
            </h2>
            <div id="help3" class="accordion-collapse collapse">
              <div class="accordion-body">
                <p>
                  Описание: классическая полносвязная нейронная сеть состоящая только из линейных
                  слоев и активаций. Решает задачу определения того, какая цифра находится на
                  поданном на вход 8 на 8 изображение. В результате (выходной слой) выдает
                  вероятности, что на картинке изображены 0, 1, 2, 3, 4, 5, 6, 7, 8 и 9
                  соответственно (сверху вниз).
                </p>
                <p>
                  Датасет: аналог датасета MNIST рукописных цифр, но с уменьшенным размером
                  изображений до 8x8. Ожидаемый результат - какая цифра изображена на картинке.
                </p>
              </div>
            </div>
          </div>

          <div class="accordion-item">
            <h2 class="accordion-header">
              <button
                class="accordion-button collapsed"
                type="button"
                data-bs-toggle="collapse"
                data-bs-target="#help4"
              >
                CNN (Сверточная нейронная сеть)
              </button>
            </h2>
            <div id="help4" class="accordion-collapse collapse">
              <div class="accordion-body">
                <p>
                  Описание: сверточная нейронная сеть, которая содержит в себе полносвязную
                  нейронную сеть и одновременно перед ней распологает специальные сверточные слои,
                  направленные для выявления закономерностей и паттернов у близко расположенных
                  пикселей изображения. Решает задачу определения того, какая цифра находится на
                  поданном на вход 8 на 8 изображение. В результате (выходной слой) выдает
                  вероятности, что на картинке изображены 0, 1, 2, 3, 4, 5, 6, 7, 8 и 9
                  соответственно (сверху вниз).
                </p>
                <p>
                  Датасет: аналог датасета MNIST рукописных цифр, но с уменьшенным размером
                  изображений до 8x8. Ожидаемый результат - какая цифра изображена на картинке.
                </p>
              </div>
            </div>
          </div>

          <div class="accordion-item">
            <h2 class="accordion-header">
              <button
                class="accordion-button collapsed"
                type="button"
                data-bs-toggle="collapse"
                data-bs-target="#help5"
              >
                GAN (Генеративная нейронная сеть)
              </button>
            </h2>
            <div id="help5" class="accordion-collapse collapse">
              <div class="accordion-body">
                <p>
                  Описание: генеративная нейронная сеть, которая состоит из двух нейронных сетей:
                  генератора и дискриминатора. <br />
                  Генератор - это нейронная сеть, которая на вход получает некоторый вектор
                  случайных чисел (шум), а на выходе должна выдать осмысленный результат. В нашем
                  случае, генератор должен на выходе получить картинку 8 на 8 сгенерированной
                  рукописной цифры. <br />
                  Дискриминатор - это нейронная сеть, которая на вход получает реальные данные и
                  сгенерированные генератором данные и должна отличить их друг от друга. Т.е. должно
                  выдать вероятность того, что поступившие на вход данные - реальные, а не
                  сгенерированные генератором. В нашем случае он получает на вход данные из датасета
                  и такое же количество сгенерированных генератором данных и определяет, какие
                  данные не "фальшивые".
                </p>
                <p>
                  Смысл генеративной нейронной сети в параллельном равномерном обучении генератора и
                  дискриминатора, так как невыгодно, чтобы кто-то из них победил. Если дискриминатор
                  будет слишком хорошо определять, какие данные сгенерированы, а какие настоящие, то
                  генератор не сможет обучиться, если дискриминатор не сможет никак отличить
                  настоящие и сгенерированные данные, то генератор остановится в совершенствовании
                  сгенерированных данных и не сможет выдать результаты лучше.
                </p>
                <p>
                  Датасет: аналог датасета MNIST рукописных цифр, но с уменьшенным размером
                  изображений до 8x8. Ожидаемый результат - какая цифра изображена на картинке.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped></style>
