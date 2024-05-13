
import torch
from tqdm import tqdm

from data import SMPL_DATA, SMAL_DATA
import utils as utils
from model256 import PoseTransformer

data_name = 'human'  # 'animal',  'human'
if (data_name == 'human'):
    dataset = SMPL_DATA(train=True, shuffle_point=True)
    num_points = 6890
    batch_size = 8
elif (data_name == 'animal'):
    dataset = SMAL_DATA(train=True, shuffle_point=True)
    num_points = 3889
    batch_size = 12
else:
    print('error: there is no dataset')
    exit(0)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = PoseTransformer(num_points=num_points)
model.cuda()
total = sum(p.numel() for p in model.parameters())
print("Total params: %.2fM" % (total / 1e6))

initial_lr = old_lr = 0.0005
optimizer_G = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.0005)

for epoch in range(0, 200):

    total_loss = 0
    total_rec_loss = 0
    total_edg_loss = 0

    if epoch > 100:
        lrd = initial_lr / 100
        old_lr = initial_lr - (epoch-101)*lrd
        new_lr = old_lr - lrd
    else:
        new_lr = old_lr
    if new_lr != old_lr:
        new_lr_G = new_lr
        for param_group in optimizer_G.param_groups:
            param_group['lr'] = new_lr_G
        print('update learning rate: %f -> %f' % (old_lr, new_lr))
        old_lr = new_lr

    for j, data in enumerate(tqdm(dataloader)):

        optimizer_G.zero_grad()
        pose_points, random_sample, gt_points, identity_points, new_face = data

        pose_points = pose_points.transpose(2, 1)  # B, 3, N
        pose_points = pose_points.cuda()

        identity_points = identity_points.transpose(2, 1)
        identity_points = identity_points.cuda()

        gt_points = gt_points.cuda()

        pointsReconstructed = model(pose_points, identity_points)

        rec_loss = torch.mean((pointsReconstructed - gt_points) ** 2)

        edg_loss = 0
        for i in range(len(random_sample)):  #
            f = new_face[i].cpu().numpy()
            v = identity_points[i].transpose(0, 1).cpu().numpy()
            edg_loss = edg_loss + utils.compute_score(pointsReconstructed[i].unsqueeze(0), f, utils.get_target(v, f, 1))

        edg_loss = edg_loss / len(random_sample)

        loss = 1000.0 * rec_loss + 0.5 * edg_loss
        loss.backward()
        optimizer_G.step()

        total_loss = total_loss + loss.item()
        total_rec_loss = total_rec_loss + 1000.0 * rec_loss.item()
        total_edg_loss = total_edg_loss + 0.5 * edg_loss.item()

    print('####################################')
    print(epoch)
    mean_loss = float((total_loss / (j + 1)))
    mean_rec_loss = float(total_rec_loss / (j + 1))
    mean_edg_loss = float(total_edg_loss / (j + 1))

    print('mean_loss', mean_loss)
    print('mean_rec_loss', mean_rec_loss)
    print('mean_edg_loss', mean_edg_loss)
    print('####################################')

    if (epoch + 1) % 5 == 0:
        save_path = './saved_model/' + data_name + '/' + str(epoch) + '.model'
        torch.save(model.state_dict(), save_path)



