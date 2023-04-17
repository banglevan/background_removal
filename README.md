# background_removal
## init project (17/04)
- service use Django (look like prev project)
- model inference on Pytorch (don't need to change env)
- put the weight `saved_model` into `runner\src`, the structure after insert model weight as
```structure
|--data
|--django_removal
|--runner
    |--migrations
    |--src
        |--models
        |--saved_models
            |--*.pth
```
