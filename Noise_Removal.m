% Load the image
img = imread('Noise.jpeg');

% Convert to grayscale if the image is RGB
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Step 1: Noise Removal Filters (Median, Gaussian, and others)
[rows, cols] = size(img);
kernel_size = 3;  % Define kernel size
pad_size = floor(kernel_size / 2);  % Padding size for filters

% Zero padding
padded_img = zeros(rows + 2 * pad_size, cols + 2 * pad_size);
padded_img(1 + pad_size:end - pad_size, 1 + pad_size:end - pad_size) = img;

% Median Filter
median_filtered = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        neighborhood = padded_img(i:i + kernel_size - 1, j:j + kernel_size - 1);
        median_filtered(i, j) = median(neighborhood(:));
    end
end
median_filtered = uint8(median_filtered);

% Arithmetic Mean Filter
arithmetic_mean_filtered = zeros(rows, cols);

for i = 1:rows
    for j = 1:cols
        % Extract the neighborhood
        neighborhood = padded_img(i:i + kernel_size - 1, j:j + kernel_size - 1);
        
        % Compute the arithmetic mean of the neighborhood
        arithmetic_mean_filtered(i, j) = mean(neighborhood(:));
    end
end

% Convert the result to uint8 format
arithmetic_mean_filtered = uint8(arithmetic_mean_filtered);

% Geometric Mean Filter
geometric_mean_filtered = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        % Ensure that the neighborhood does not exceed the bounds of the image
        neighborhood = padded_img(i:min(i + kernel_size - 1, end), j:min(j + kernel_size - 1, end));
        geometric_mean_filtered(i, j) = exp(mean(log(double(neighborhood(:)))));
    end
end
geometric_mean_filtered = uint8(geometric_mean_filtered);

% Harmonic Mean Filter
harmonic_mean_filtered = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        % Ensure that the neighborhood does not exceed the bounds of the image
        neighborhood = padded_img(i:min(i + kernel_size - 1, end), j:min(j + kernel_size - 1, end));
        harmonic_mean_filtered(i, j) = 1 / mean(1 ./ double(neighborhood(:)));
    end
end
harmonic_mean_filtered = uint8(harmonic_mean_filtered);

% Contraharmonic Mean Filter (Q = -1)
contraharmonic_mean_filtered = zeros(rows, cols);
Q = -1;
for i = 1:rows
    for j = 1:cols
        % Ensure that the neighborhood does not exceed the bounds of the image
        neighborhood = padded_img(i:min(i + kernel_size - 1, end), j:min(j + kernel_size - 1, end));
        contraharmonic_mean_filtered(i, j) = sum(neighborhood(:).^(Q + 1)) / sum(neighborhood(:).^Q);
    end
end
contraharmonic_mean_filtered = uint8(contraharmonic_mean_filtered);

% Midpoint Filter
midpoint_filtered = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        % Ensure that the neighborhood does not exceed the bounds of the image
        neighborhood = padded_img(i:min(i + kernel_size - 1, end), j:min(j + kernel_size - 1, end));
        midpoint_filtered(i, j) = (min(neighborhood(:)) + max(neighborhood(:))) / 2;
    end
end
midpoint_filtered = uint8(midpoint_filtered);

% Max & Min Filter
max_filtered = zeros(rows, cols);
min_filtered = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        % Ensure that the neighborhood does not exceed the bounds of the image
        neighborhood = padded_img(i:min(i + kernel_size - 1, end), j:min(j + kernel_size - 1, end));
        max_filtered(i, j) = max(neighborhood(:));
        min_filtered(i, j) = min(neighborhood(:));
    end
end
max_filtered = uint8(max_filtered);
min_filtered = uint8(min_filtered);

% Alpha Trimmed Mean Filter (alpha = 1)
alpha = 1;
alpha_trimmed_filtered = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        % Ensure that the neighborhood does not exceed the bounds of the image
        neighborhood = padded_img(i:min(i + kernel_size - 1, end), j:min(j + kernel_size - 1, end));
        sorted_neighborhood = sort(neighborhood(:));
        trimmed_values = sorted_neighborhood(1 + alpha:end - alpha);
        alpha_trimmed_filtered(i, j) = mean(trimmed_values);
    end
end
alpha_trimmed_filtered = uint8(alpha_trimmed_filtered);

% Adaptive Median Filter
adaptive_median_filtered = zeros(rows, cols);

% Maximum window size
S_max = 7;  % Set a reasonable maximum window size (odd value)
initial_kernel_size = 3;  % Start with a small kernel size

for i = 1:rows
    for j = 1:cols
        kernel_size = initial_kernel_size;
        output_value = 0;
        while kernel_size <= S_max
            % Ensure the neighborhood does not exceed bounds
            row_start = max(i - pad_size, 1);
            row_end = min(i + pad_size, size(padded_img, 1));
            col_start = max(j - pad_size, 1);
            col_end = min(j + pad_size, size(padded_img, 2));
            
            % Extract the neighborhood
            neighborhood = padded_img(row_start:row_end, col_start:col_end);
            
            % Calculate statistics for the neighborhood
            z_min = min(neighborhood(:));
            z_max = max(neighborhood(:));
            z_med = median(neighborhood(:));
            z_xy = padded_img(i + pad_size, j + pad_size);
            
            % Level A
            A1 = z_med - z_min;
            A2 = z_med - z_max;
            if A1 > 0 && A2 < 0
                % Go to Level B
                B1 = z_xy - z_min;
                B2 = z_xy - z_max;
                if B1 > 0 && B2 < 0
                    output_value = z_xy;
                else
                    output_value = z_med;
                end
                break;  % Exit the loop
            else
                % Increase window size
                kernel_size = kernel_size + 2;
            end
        end
        
        % If no output value is set (window size exceeded), output z_med
        if output_value == 0
            output_value = z_med;
        end
        
        % Assign the output value to the filtered image
        adaptive_median_filtered(i, j) = output_value;
    end
end

% Convert the final output to uint8
adaptive_median_filtered = uint8(adaptive_median_filtered);


% Step 2: Apply smoothing filter (9x9 Mean Filter)
filter_size = 9;  % 9x9 filter
filter = (1/81) * ones(filter_size, filter_size);  % Define smoothing filter

% Get image dimensions
[rows, cols] = size(img);

% Calculate padding size for convolution
pad_size = floor(filter_size / 2);

% Pad the input image with zeros
padded_image = zeros(rows + 2 * pad_size, cols + 2 * pad_size);
padded_image(pad_size + 1:end - pad_size, pad_size + 1:end - pad_size) = img;

% Initialize the final output image
final_output_image = zeros(rows, cols);

% Perform convolution manually for smoothing
for i = 1:rows
    for j = 1:cols
        % Extract the region of interest (local neighborhood)
        region = padded_image(i:i + 2 * pad_size, j:j + 2 * pad_size);
        
        % Apply the smoothing filter (element-wise multiplication and summation)
        value = sum(sum(region .* filter));
        
        % Assign the result to the final output image
        final_output_image(i, j) = value;
    end
end

% Convert the final output image to uint8
final_output_image = uint8(final_output_image);

% Display images for all filtered images in one window
figure;

subplot(4, 3, 1);
imshow(img);
title('Original Image');

subplot(4, 3, 2);
imshow(median_filtered);
title('Median Filtered Image');

subplot(4, 3, 3);
imshow(geometric_mean_filtered);
title('Geometric Mean Filtered Image');

subplot(4, 3, 4);
imshow(harmonic_mean_filtered);
title('Harmonic Mean Filtered Image');

subplot(4, 3, 5);
imshow(contraharmonic_mean_filtered);
title('Contraharmonic Mean Filtered Image');

subplot(4, 3, 6);
imshow(midpoint_filtered);
title('Midpoint Filtered Image');

subplot(4, 3, 7);
imshow(max_filtered);
title('Max Filtered Image');

subplot(4, 3, 8);
imshow(min_filtered);
title('Min Filtered Image');

subplot(4, 3, 9);
imshow(adaptive_median_filtered);
title('Adaptive Median Filtered Image');

subplot(4, 3, 10);
imshow(alpha_trimmed_filtered);
title('alpha trimmed filtered Image');

subplot(4, 3, 11);
imshow(arithmetic_mean_filtered);
title('arithmetic mean filtered Image');

subplot(4, 3, 12);
imshow(final_output_image);
title('Average mean Image');
