import streamlit_authenticator as stauth

# Generate hash satu per satu (lebih jelas)
print("\n" + "="*70)
print("HASHED PASSWORDS")
print("="*70 + "\n")

# Admin
admin_hash = stauth.Hasher(['admin123']).generate()
print("1. ADMIN")
print(f"   Password: admin123")
print(f"   Hash: {admin_hash[0]}")
print()

# Apoteker
apoteker_hash = stauth.Hasher(['apoteker123']).generate()
print("2. APOTEKER")
print(f"   Password: apoteker123")
print(f"   Hash: {apoteker_hash[0]}")
print()

# Dokter
dokter_hash = stauth.Hasher(['dokter123']).generate()
print("3. DOKTER")
print(f"   Password: dokter123")
print(f"   Hash: {dokter_hash[0]}")
print()

print("="*70)
print("Copy hash di atas ke config.yaml")
print("="*70)