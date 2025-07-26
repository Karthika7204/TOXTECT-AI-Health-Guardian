from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('calories/',views.calories, name="calories"),
    path('auth/',views.auth_page,name="auth_page"),
    path('signup/', views.user_signup, name='signup'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('dashboard/',views.dashboard,name="dashboard"),
    path('scan/',views.scan,name="scan"),
    path('process/',views.process_image,name="process_image"),
    path('health/', views.health, name='health'),
    path('bmi/',views.bmi_predictor,name="bmi_predictor"),
    path('vision_check',views.vision_check,name="vision_check"),
    path('tablet/',views.tablet_tracker,name="tablet_tracker"),
    path('save_tablets/', views.save_tablets, name='save_tablets'),
    path('mental_health/', views.start_assessment, name='start_assessment'),
    path('heart_attack_predition/', views.predict_heart_attack, name='predict_heart_attack'),
    
]