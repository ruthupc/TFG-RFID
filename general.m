%% Cantidad de cirugías/día

%Cargar manual documento cirugias.xlsx

% Extraer columnas de interés
fechas = cirugias{:,1}; % Primera columna (fechas)
duraciones = cirugias{:,2}; % Segunda columna (duración de las cirugías)

% Convertir a formato datetime
fechas = datetime(fechas);

% Extraer solo la parte de la fecha
fechas = dateshift(fechas, 'start', 'day');

% Crear histograma para la distribución de cirugías por día
figure(1);
histogram(fechas, 'BinMethod', 'day');
title('Distribución de cirugías por día');
xlabel('Día');
ylabel('Número de cirugías');


% Contar el número de cirugías por día 
[unique_days, ~, day_indices] = unique(fechas);
cirugias_por_dia = accumarray(day_indices, 1);

% Crear el histograma de las cantidades de cirugías diarias
figure;
histogram(cirugias_por_dia);
title('Distribución de la cantidad de cirugías diarias');
xlabel('Cirugías diarias');
ylabel('Frecuencia');

% Ajustar varias distribuciones a los datos de cantidades de cirugías diarias
pd_poisson = fitdist(cirugias_por_dia, 'Poisson');
pd_normal = fitdist(cirugias_por_dia, 'Normal');
pd_binomial = fitdist(cirugias_por_dia, 'Binomial', 'NTrials', max(cirugias_por_dia));
pd_exponential = fitdist(cirugias_por_dia, 'Exponential');

% Calcular el logaritmo de la función de densidad de probabilidad (logPDF) de cada distribución en los puntos de los datos
log_pdf_poisson = log(pdf(pd_poisson, cirugias_por_dia));
log_pdf_normal = log(pdf(pd_normal, cirugias_por_dia));
log_pdf_binomial = log(pdf(pd_binomial, cirugias_por_dia));
log_pdf_exponential = log(pdf(pd_exponential, cirugias_por_dia));

% Calcular el AIC de cada distribución para la elección de cual se ajusta
% mejor
num_params_poisson = pd_poisson.NumParameters;
num_params_normal = pd_normal.NumParameters;
num_params_binomial = pd_binomial.NumParameters;
num_params_exponential = pd_exponential.NumParameters;

n = numel(cirugias_por_dia); % Número de observaciones
aic_poisson = 2 * num_params_poisson - 2 * sum(log_pdf_poisson);
aic_normal = 2 * num_params_normal - 2 * sum(log_pdf_normal);
aic_binomial = 2 * num_params_binomial - 2 * sum(log_pdf_binomial);
aic_exponential = 2 * num_params_exponential - 2 * sum(log_pdf_exponential);

% Comparar los AICs para seleccionar la mejor distribución
[aic_values, best_fit_index] = min([aic_poisson; aic_normal; aic_binomial; aic_exponential]);
best_fit_distributions = {'Poisson', 'Normal', 'Binomial', 'Exponential'};
best_fit = best_fit_distributions{best_fit_index};

% Mostrar los resultados
disp(['Mejor ajuste: ', best_fit]);
disp(['AIC Poisson: ', num2str(aic_poisson)]);
disp(['AIC Normal: ', num2str(aic_normal)]);
disp(['AIC Binomial: ', num2str(aic_binomial)]);
disp(['AIC Exponential: ', num2str(aic_exponential)]);


% Obtener la función de la mejor distribución numéricamente
if strcmp(best_fit, 'Poisson')
    best_fit_function = @(x) pdf(pd_poisson, x);
elseif strcmp(best_fit, 'Normal')
    best_fit_function = @(x) pdf(pd_normal, x);
elseif strcmp(best_fit, 'Binomial')
    best_fit_function = @(x) pdf(pd_binomial, x);
elseif strcmp(best_fit, 'Exponential')
    best_fit_function = @(x) pdf(pd_exponential, x);
end

% Mostrar la función de la mejor distribución
disp(['La función de la mejor distribución (' best_fit ') es:']);
disp(func2str(best_fit_function));

% Obtener los parámetros mu y sigma de la distribución normal ajustada
mu_normal = pd_normal.mu;
sigma_normal = pd_normal.sigma;

% Mostrar los valores de mu y sigma
disp(['Parámetros de la distribución Normal ajustada:']);
disp(['Mu (media): ', num2str(mu_normal)]);
disp(['Sigma (desviación estándar): ', num2str(sigma_normal)]);

% Crear el histograma de las cantidades de cirugías diarias
figure;
histogram(cirugias_por_dia, 'Normalization', 'pdf');
hold on;

% Ajustar varias distribuciones a los datos de cantidades de cirugías diarias
pd_poisson = fitdist(cirugias_por_dia, 'Poisson');
pd_normal = fitdist(cirugias_por_dia, 'Normal');
pd_binomial = fitdist(cirugias_por_dia, 'Binomial', 'NTrials', max(cirugias_por_dia));


% Graficar la PDF de cada distribución ajustada
x_values = min(cirugias_por_dia):max(cirugias_por_dia);
y_values_poisson = pdf(pd_poisson, x_values);
y_values_normal = pdf(pd_normal, x_values);
y_values_binomial = pdf(pd_binomial, x_values);

plot(x_values, y_values_poisson, 'y-', 'LineWidth', 2);
plot(x_values, y_values_normal, 'r-', 'LineWidth', 2);
plot(x_values, y_values_binomial, 'b-', 'LineWidth', 2);


% Gráfico títulos y leyenda
title('Distribución de la cantidad de cirugías diarias');
xlabel('Cantidad de cirugías');
ylabel('Densidad de probabilidad');
legend('Histograma de densidad de probabilidad', 'Poisson', 'Normal', 'Binomial', 'Exponential');
hold off;

%% Duración de cirugías
% Ordenar el vector de duraciones en orden descendente
duraciones_ordenadas = sort(duraciones, 'descend');

% Calcular el número de elementos a eliminar (10% de los datos)
num_elementos_a_eliminar = round(0.1 * numel(duraciones_ordenadas));

% Eliminar los primeros elementos del vector ordenado
duraciones_filtradas = duraciones_ordenadas(num_elementos_a_eliminar + 1:end);

duraciones=duraciones_filtradas;

% Ajustar varias distribuciones a los datos de duración de cirugías
pd_normal_duraciones = fitdist(duraciones, 'Normal');
pd_exponential_duraciones = fitdist(duraciones, 'Exponential');
pd_poisson_duraciones = fitdist(duraciones, 'Poisson');

% Calcular el logaritmo de la función de densidad de probabilidad (logPDF) de cada distribución en los puntos de los datos
log_pdf_normal_duraciones = log(pdf(pd_normal_duraciones, duraciones));
log_pdf_exponential_duraciones = log(pdf(pd_exponential_duraciones, duraciones));
log_pdf_poisson_duraciones = log(pdf(pd_poisson_duraciones, duraciones));

% Calcular el AIC de cada distribución
num_params_normal = pd_normal_duraciones.NumParameters;
num_params_exponential = pd_exponential_duraciones.NumParameters;
num_params_poisson = pd_poisson_duraciones.NumParameters;

n = numel(duraciones); % Número de observaciones
aic_normal_duraciones = 2 * num_params_normal - 2 * sum(log_pdf_normal_duraciones);
aic_exponential_duraciones = 2 * num_params_exponential - 2 * sum(log_pdf_exponential_duraciones);
aic_poisson_duraciones = 2 * num_params_poisson - 2 * sum(log_pdf_poisson_duraciones);

% Comparar los AICs para seleccionar la mejor distribución
[aic_values, best_fit_index] = min([aic_normal_duraciones; aic_exponential_duraciones; aic_poisson_duraciones]);
best_fit_distributions = {'Normal', 'Exponential', 'Poisson'};
best_fit = best_fit_distributions{best_fit_index};

% Mostrar los resultados
disp(['Mejor ajuste: ', best_fit]);
disp(['AIC Normal: ', num2str(aic_normal_duraciones)]);
disp(['AIC Exponential: ', num2str(aic_exponential_duraciones)]);
disp(['AIC Poisson: ', num2str(aic_poisson_duraciones)]);


%Graficar
% La distribución Beta requiere que los datos estén en el rango (0,1)
% Normalizamos los datos para ajustarlos a la distribución Beta
min_dur = min(duraciones);
max_dur = max(duraciones);
norm_duraciones = (duraciones - min_dur) / (max_dur - min_dur);
pd_beta_duraciones = fitdist(norm_duraciones, 'Beta');

% Obtener la función de densidad de probabilidad (PDF) de las distribuciones ajustadas
x_values_duraciones = linspace(min_dur, max_dur, 1000);
y_values_normal_duraciones = pdf(pd_normal_duraciones, x_values_duraciones);
y_values_exponential_duraciones = pdf(pd_exponential_duraciones, x_values_duraciones);
y_values_poisson_duraciones = pdf(pd_poisson_duraciones, x_values_duraciones);
y_values_beta_duraciones = pdf(pd_beta_duraciones, (x_values_duraciones - min_dur) / (max_dur - min_dur));

% Graficar la PDF de cada distribución ajustada
% Calcular los límites de los bins del histograma (de 3 en 3 minutos)
bin_edges_duraciones = min_dur:2:max_dur;

% Graficar el histograma con los bins de 3 en 3 minutos
figure;
histogram(duraciones, bin_edges_duraciones, 'Normalization', 'pdf');
hold on;
plot(x_values_duraciones, y_values_normal_duraciones, 'r-', 'LineWidth', 2);
plot(x_values_duraciones, y_values_exponential_duraciones, 'g-', 'LineWidth', 2);
%plot(x_values_duraciones, y_values_poisson_duraciones, 'b-', 'LineWidth',
%2); no es suficientemente bueno
%plot(x_values_duraciones, y_values_beta_duraciones, 'm-', 'LineWidth', 2);
%no es suficientemente bueno

% Gráfico títulos y leyenda
title('Distribución de la duración de las cirugías');
xlabel('Duración de las cirugías');
ylabel('Densidad de probabilidad');
legend('Histograma de densidad', 'Normal', 'Exponential', 'Poisson', 'Beta');
hold off;

% Ajustar una distribución exponencial a los datos de duración de cirugías
pd_exponential_duraciones = fitdist(duraciones, 'Exponential');

% Calcular y mostrar los parámetros de la distribución exponencial
mu_exponential = pd_exponential_duraciones.mu; % Parámetro mu
scale_exponential = 1 / mu_exponential; % Parámetro de escala (equivalente a lambda)

disp(['El valor de mu para la distribución exponencial es: ', num2str(mu_exponential)]);
disp(['El valor de escala para la distribución exponencial es: ', num2str(scale_exponential)]);

% Ajustar una distribución normal a los datos filtrados
pd_normal_duraciones_filtradas = fitdist(duraciones_filtradas, 'Normal');

% Obtener los parámetros de la distribución normal ajustada
mu_normal_filtrada = pd_normal_duraciones_filtradas.mu; % Media
sigma_normal_filtrada = pd_normal_duraciones_filtradas.sigma; % Desviación estándar

% Mostrar los valores de mu y sigma
disp(['Parámetros de la distribución Normal ajustada a los datos filtrados:']);
disp(['Mu (media): ', num2str(mu_normal_filtrada)]);
disp(['Sigma (desviación estándar): ', num2str(sigma_normal_filtrada)]);

%% Registro
% Cargar los datos 
data = readmatrix('tiempos_recepcion.xlsx');

% Extraer los tiempos de recepción
tiempos_recepcion = data(:, 2);

% Definir el ancho de los contenedores del histograma (en segundos)
bin_width_seconds = 10;

% Calcular el número de contenedores basado en el ancho deseado
bin_edges = min(tiempos_recepcion):bin_width_seconds:max(tiempos_recepcion);

% Ajustar varias distribuciones a los datos
pd_normal = fitdist(tiempos_recepcion, 'Normal');
pd_exponential = fitdist(tiempos_recepcion, 'Exponential');
pd_poisson = fitdist(tiempos_recepcion, 'Poisson');

% Calcular el AIC de cada distribución
log_likelihood_normal = sum(log(pdf(pd_normal, tiempos_recepcion)));
log_likelihood_exponential = sum(log(pdf(pd_exponential, tiempos_recepcion)));
log_likelihood_poisson = sum(log(pdf(pd_poisson, tiempos_recepcion)));

num_params_normal = pd_normal.NumParameters;
num_params_exponential = pd_exponential.NumParameters;
num_params_poisson = pd_poisson.NumParameters;

n = numel(tiempos_recepcion); % Número de observaciones
aic_normal = 2 * num_params_normal - 2 * log_likelihood_normal;
aic_exponential = 2 * num_params_exponential - 2 * log_likelihood_exponential;
aic_poisson = 2 * num_params_poisson - 2 * log_likelihood_poisson;

% Mostrar los resultados
disp('Resultados del ajuste de distribuciones:');
disp(['AIC Normal: ', num2str(aic_normal)]);
disp(['AIC Exponential: ', num2str(aic_exponential)]);
disp(['AIC Poisson: ', num2str(aic_poisson)]);

% Obtener los parámetros de la distribución normal
mu_normal = pd_normal.mu;
sigma_normal = pd_normal.sigma;
disp(['Mu de la distribución normal: ', num2str(mu_normal)]);
disp(['Sigma de la distribución normal: ', num2str(sigma_normal)]);

% Comparar los AICs para seleccionar la mejor distribución
[aic_values, best_fit_index] = min([aic_normal, aic_exponential, aic_poisson]);
best_fit_distributions = {'Normal', 'Exponential', 'Poisson'};
best_fit = best_fit_distributions{best_fit_index};

% Mostrar los resultados
disp(['Mejor ajuste: ', best_fit]);
disp(['AIC ', best_fit, ': ', num2str(aic_values)]);

% Obtener la función de densidad de probabilidad (PDF) de la mejor distribución
if strcmp(best_fit, 'Normal')
    best_fit_function = @(x) pdf(pd_normal, x);
elseif strcmp(best_fit, 'Exponential')
    best_fit_function = @(x) pdf(pd_exponential, x);
elseif strcmp(best_fit, 'Poisson')
    best_fit_function = @(x) pdf(pd_poisson, x);
end

% Graficar el histograma con las distribuciones encima
figure;
histogram(tiempos_recepcion, bin_edges, 'Normalization', 'pdf');
hold on;
x_values = linspace(min(tiempos_recepcion), max(tiempos_recepcion), 1000);
plot(x_values, pdf(pd_normal, x_values), 'r-', 'LineWidth', 2);
plot(x_values, pdf(pd_exponential, x_values), 'g-', 'LineWidth', 2);
%plot(x_values, pdf(pd_poisson, x_values), 'b-', 'LineWidth', 2); ya que no
%se ajusta bien

% Gráfico títulos y leyenda
title('Distribución de tiempos de recepción');
xlabel('Tiempos de recepción');
ylabel('Densidad de probabilidad');
legend('Datos', 'Normal', 'Exponential', 'Poisson');
hold off;


%% Paquetes
% Cantidad total de productos por tipo de paquete
total_products_5 = 48;
total_products_6 = 12;
total_products_4 = 10;

% Calcular el número de paquetes
num_packs_5 = floor(total_products_5 / 5);
num_packs_6 = floor(total_products_6 / 6);
num_packs_4 = floor(total_products_4 / 4);

% Crear un vector con el tamaño de cada paquete
packages = [repmat(5, 1, num_packs_5), repmat(6, 1, num_packs_6), repmat(4, 1, num_packs_4)];

% Crear el histograma
figure;
histogram(packages, 'BinMethod', 'integers');
xlabel('Tamaño del Paquete');
ylabel('Frecuencia');
title('Distribución de Productos por Tamaño de Paquete');

% Calcular y devolver media y la desviación estándar
mu = mean(packages);
sigma = std(packages);
disp(['Media (mu): ', num2str(mu)]);
disp(['Desviación Estándar (sigma): ', num2str(sigma)]);

% Crear el histograma
figure;
histogram(packages, 'Normalization', 'pdf', 'BinMethod', 'integers');
hold on;

% Generar datos para la distribución normal
x = min(packages):0.1:max(packages);
pdf_normal = normpdf(x, mu, sigma);

% Plotear la distribución normal
plot(x, pdf_normal, 'r-', 'LineWidth', 2);
hold off;

% Etiquetas y título
xlabel('Tamaño del Paquete');
ylabel('Densidad de Probabilidad');
title('Distribución de Productos por Tamaño de Paquete con Distribución Normal');
legend('Datos', 'Normal');

%% Mantenimiento
% Cargar el archivo de Excel
data = readtable('mantenimiento.xlsx');

% Extraer las fechas de mantenimiento
fechas = data{:,1}; % Asumiendo que las fechas están en la primera columna

% Convertir a formato datetime si no lo están
fechas = datetime(fechas);
fechas = fechas(1:419, :);

% Extraer solo la parte de la fecha (sin horas)
fechas = dateshift(fechas, 'start', 'day');

% Crear histograma para la distribución de mantenimientos por semana
figure;
histogram(fechas, 'BinMethod', 'week');
title('Distribución de mantenimientos por semana');
xlabel('Semana');
ylabel('Número de mantenimientos');

% Contar el número de mantenimientos por semana
[~, ~, week_indices] = unique(week(fechas));
mantenimientos_por_semana = accumarray(week_indices, 1);

% Crear el histograma de las cantidades de mantenimientos semanales
figure;
histogram(mantenimientos_por_semana);
title('Distribución de la cantidad de mantenimientos semanales');
xlabel('Mantenimientos semanales');
ylabel('Frecuencia');

% Ajustar varias distribuciones a los datos de cantidades de mantenimientos semanales
pd_poisson = fitdist(mantenimientos_por_semana, 'Poisson');
pd_normal = fitdist(mantenimientos_por_semana, 'Normal');
pd_binomial = fitdist(mantenimientos_por_semana, 'Binomial', 'NTrials', max(mantenimientos_por_semana));
pd_exponential = fitdist(mantenimientos_por_semana, 'Exponential');

% Calcular el logaritmo de la función de densidad de probabilidad (logPDF) de cada distribución en los puntos de los datos
log_pdf_poisson = log(pdf(pd_poisson, mantenimientos_por_semana));
log_pdf_normal = log(pdf(pd_normal, mantenimientos_por_semana));
log_pdf_binomial = log(pdf(pd_binomial, mantenimientos_por_semana));
log_pdf_exponential = log(pdf(pd_exponential, mantenimientos_por_semana));

% Calcular el AIC de cada distribución
num_params_poisson = pd_poisson.NumParameters;
num_params_normal = pd_normal.NumParameters;
num_params_binomial = pd_binomial.NumParameters;
num_params_exponential = pd_exponential.NumParameters;

n = numel(mantenimientos_por_semana); % Número de observaciones
aic_poisson = 2 * num_params_poisson - 2 * sum(log_pdf_poisson);
aic_normal = 2 * num_params_normal - 2 * sum(log_pdf_normal);
aic_binomial = 2 * num_params_binomial - 2 * sum(log_pdf_binomial);
aic_exponential = 2 * num_params_exponential - 2 * sum(log_pdf_exponential);

% Comparar los AICs para seleccionar la mejor distribución
[aic_values, best_fit_index] = min([aic_poisson; aic_normal; aic_binomial; aic_exponential]);
best_fit_distributions = {'Poisson', 'Normal', 'Binomial', 'Exponential'};
best_fit = best_fit_distributions{best_fit_index};

% Mostrar los resultados
disp(['Mejor ajuste: ', best_fit]);
disp(['AIC Poisson: ', num2str(aic_poisson)]);
disp(['AIC Normal: ', num2str(aic_normal)]);
disp(['AIC Binomial: ', num2str(aic_binomial)]);
disp(['AIC Exponential: ', num2str(aic_exponential)]);

% Obtener la función de la mejor distribución numéricamente
if strcmp(best_fit, 'Poisson')
    best_fit_function = @(x) pdf(pd_poisson, x);
elseif strcmp(best_fit, 'Normal')
    best_fit_function = @(x) pdf(pd_normal, x);
elseif strcmp(best_fit, 'Binomial')
    best_fit_function = @(x) pdf(pd_binomial, x);
elseif strcmp(best_fit, 'Exponential')
    best_fit_function = @(x) pdf(pd_exponential, x);
end

% Mostrar la función de la mejor distribución
disp(['La función de la mejor distribución (' best_fit ') es:']);
disp(func2str(best_fit_function));

% Crear el histograma de las cantidades de mantenimientos semanales
figure;
histogram(mantenimientos_por_semana, 'Normalization', 'pdf');
hold on;

% Graficar la PDF de cada distribución ajustada
x_values = min(mantenimientos_por_semana):max(mantenimientos_por_semana);
y_values_poisson = pdf(pd_poisson, x_values);
y_values_normal = pdf(pd_normal, x_values);
y_values_binomial = pdf(pd_binomial, x_values);
y_values_exponencial = pdf(pd_exponential, x_values);

plot(x_values, y_values_poisson, 'y-', 'LineWidth', 2);
plot(x_values, y_values_normal, 'r-', 'LineWidth', 2);
plot(x_values, y_values_binomial, 'b-', 'LineWidth', 2);
plot(x_values, y_values_exponencial, 'g-', 'LineWidth', 2);

% Configurar el gráfico
title('Distribución de la cantidad de mantenimientos semanales');
xlabel('Cantidad de mantenimientos');
ylabel('Densidad de probabilidad');
legend('Histograma de densidad de probabilidad', 'Poisson', 'Normal', 'Binomial', 'Exponential');
hold off;

% Obtener la media (lambda) de la distribución de Poisson ajustada
lambda_poisson = pd_poisson.lambda;
disp(['El valor de la media (lambda) de la distribución de Poisson ajustada es: ', num2str(lambda_poisson)]);

%% Evolución anual mantenimientos
% Cargar el archivo de Excel
data = readtable('mantenimiento.xlsx');

% Extraer las fechas de mantenimiento
fechas = data{:,2}; % Asumiendo que las fechas están en la primera columna

% Convertir a formato datetime si no lo están
fechas = datetime(fechas);

% Extraer solo la parte de la fecha (sin horas)
fechas = dateshift(fechas, 'start', 'year');

% Crear histograma para la distribución de mantenimientos por semana
figure;
histogram(fechas, 'BinMethod', 'year');
title('Distribución de mantenimientos por año');
xlabel('Año');
ylabel('Número de mantenimientos');


%% Cantidad de envios y de cuantas cajas. SIN RFID

% Datos proporcionados
datos = [
    12, 3; 10, 1; 1, 1; 5, 4; 8, 1; 2, 1; 5, 2; 6, 2;
    12, 2; 10, 2; 10, 4; 8, 2; 10, 2; 5, 1; 1, 1
];

% Separar datos en cajas y días
cajas = datos(:, 1);
dias = datos(:, 2);

% Calcular la cantidad total de cajas enviadas por cada intervalo de días
[unicosDias, ~, idx] = unique(dias);
totalCajasPorDia = accumarray(idx, cajas);

% Calcular la frecuencia de envíos para cada intervalo de días
frecuenciaEnvios = accumarray(idx, 1);

% Crear gráficos
figure;

% Gráfico para la cantidad total de cajas enviadas por intervalo de días
subplot(1, 2, 1);
bar(unicosDias, totalCajasPorDia, 'FaceColor', 'blue', 'EdgeColor', 'black');
xlabel('Días');
ylabel('Cantidad Total de Cajas');
title('Cantidad Total de Cajas Enviadas por Intervalo de Días');

% Gráfico para la frecuencia de envíos por intervalo de días
subplot(1, 2, 2);
bar(unicosDias, frecuenciaEnvios, 'FaceColor', 'cyan', 'EdgeColor', 'black');
xlabel('Días');
ylabel('Frecuencia de Envíos');
title('Frecuencia de Envíos por Intervalo de Días');






