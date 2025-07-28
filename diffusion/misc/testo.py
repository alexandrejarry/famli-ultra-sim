def forward(self, x, condition=None):
    # Example: Using all components in a conditional model
    x = self.encoder(x)
    if condition is not None:
        condition = self.condition_processor(condition)
        x = torch.cat((x, condition), dim=1)
    x = self.decoder(x)
    return x

