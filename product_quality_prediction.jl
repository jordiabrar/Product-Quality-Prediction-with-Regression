############################################################
# Prediksi Kualitas Produk dengan Regresi
#
# Proyek ini menggunakan MLJ.jl dan MLJLinearModels untuk
# membangun model regresi linear yang memprediksi kualitas
# produk berdasarkan fitur-fitur produk.
#
# Fitur:
# - price          : Harga produk
# - review_count   : Jumlah review produk
# - average_rating : Rata-rata rating produk
# - discount       : Diskon produk dalam persen
#
# Target:
# - quality        : Kualitas produk (nilai kontinu)
############################################################

using DataFrames
using Random
using Statistics
using MLJ
using MLJLinearModels
using Plots

# Set seed untuk reproduksibilitas
Random.seed!(123)

# Jumlah sampel data
n = 300

# Generate data sintetis
price = rand(Normal(50, 10), n)           # Harga produk
review_count = rand(50:500, n)              # Jumlah review
average_rating = rand(Uniform(1, 5), n)     # Rata-rata rating produk
discount = rand(Uniform(0, 30), n)          # Diskon produk (dalam persen)

# Rumus kualitas produk (simulasi):
# quality = 5 + (-0.02 * price) + (0.005 * review_count) +
#           (1.2 * average_rating) + (0.1 * discount) + noise
noise = rand(Normal(0, 0.5), n)
quality = 5 .+ (-0.02).*price .+ 0.005.*review_count .+ 1.2.*average_rating .+ 0.1.*discount .+ noise

# Buat DataFrame
df = DataFrame(price=price,
               review_count=review_count,
               average_rating=average_rating,
               discount=discount,
               quality=quality)

println("Contoh data:")
first(df, 5) |> println

# Pisahkan data menjadi fitur (X) dan target (y)
X = select(df, Not(:quality))
y = df.quality

# Bagi data menjadi set pelatihan (80%) dan set pengujian (20%)
train_indices, test_indices = partition(eachindex(y), 0.8, shuffle=true)

X_train = X[train_indices, :]
y_train = y[train_indices]

X_test = X[test_indices, :]
y_test = y[test_indices]

# Definisikan model regresi linear menggunakan MLJLinearModels
model = LinearRegressor()

# Bungkus model ke dalam machine
mach = machine(model, X_train, y_train)

# Latih model
fit!(mach)

# Lakukan prediksi pada set pengujian
y_pred = predict(mach, X_test)

# Hitung metrik evaluasi
rmse = sqrt(mean((y_test .- y_pred).^2))
r2 = 1 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

println("\nEvaluasi Model:")
println("RMSE: ", round(rmse, digits=4))
println("R-squared: ", round(r2, digits=4))

# Plot perbandingan antara nilai aktual dan prediksi
scatter(y_test, y_pred,
    xlabel="Kualitas Produk Aktual",
    ylabel="Kualitas Produk Prediksi",
    title="Perbandingan Aktual vs Prediksi",
    legend=false)
plot!(identity, label="Ideal Fit", line=:dash)

savefig("pred_vs_actual.png")
println("\nPlot perbandingan telah disimpan sebagai 'pred_vs_actual.png'.")
