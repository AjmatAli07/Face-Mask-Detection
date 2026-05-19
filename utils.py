from torchvision import transforms

# Image transformations
transform = transforms.Compose([

    # Resize all images
    transforms.Resize((128,128)),

    # Random left-right flip
    transforms.RandomHorizontalFlip(),

    # Small rotation for angle variation
    transforms.RandomRotation(15),

    # Simulate dark/bright environments
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.3
    ),

    # Convert image to tensor
    transforms.ToTensor(),

    # Normalize
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])