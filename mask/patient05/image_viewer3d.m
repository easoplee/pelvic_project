%load mri
V = tiffreadVolume('patient5_femur.tif');
Ds = smooth3(V);
x = Ds(:,1);
[f,v] = isosurface(Ds);
[f2, v2, c] = isocaps(V);
f3 = [f ; f2+length(v(:,1))];
v3 = [v ; v2];
TR=triangulation(f3,v3);
trisurf(TR, 'FaceColor', [0.5,0.5,0.5],...
    'EdgeColor', [0.6,0.6,0.6])
mycolors = [1 1 1];
colormap(mycolors);
daspect([1,1,0.2])                   
lightangle(-15,-23); 
%set(gcf,'Renderer','zbuffer'); lighting phong
lighting gouraud
axis off

stlwrite(TR,'test_femur.stl');